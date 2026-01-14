import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from multimodal_transformers.model.tabular_config import TabularConfig
from multimodal_transformers.model.tabular_combiner import TabularFeatCombiner
from registry import MODEL_REGISTRY


class Reducer(nn.Module):
    def __init__(self):
        super().__init__()
        # Transformer层配置
        self.transformer_hidden_size = 128
        self.transformer_num_heads = 4
        self.transformer_num_layers = 2
        self.transformer_dropout = 0.15
        
        # Transformer编码器层
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_hidden_size,
            nhead=self.transformer_num_heads,
            dim_feedforward=self.transformer_hidden_size * 4,
            dropout=self.transformer_dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=self.transformer_num_layers
        )
        
        # Bi-GRU层配置
        self.hidden_size = 128  # 双向GRU的隐藏层大小，总输出维度为256
        self.bi_gru = nn.GRU(
            input_size=self.transformer_hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=True,
            dropout=0.15,
            batch_first=True  # 输入格式为(batch_size, seq_len, d_model)
        )
        
        # 调整输出层以匹配新的隐藏层大小
        self.output_layer = nn.Linear(self.hidden_size * 2, 128)

    def forward(self, x, attention_mask):
        # Transformer输入格式为(seq_len, batch_size, d_model)
        # 转换为(batch_size, seq_len, d_model)
        x = x.transpose(0, 1)
        
        # 调整attention_mask的形状以匹配batch_first格式
        if attention_mask.size(0) != x.size(0) or attention_mask.size(1) != x.size(1):
            attention_mask = attention_mask.transpose(0, 1)
        
        # 保存原始输入的序列长度
        original_seq_len = x.size(1)
        
        # 计算有效序列长度（非填充部分的长度）
        seq_lengths = (~attention_mask).sum(dim=1)
        
        # 先经过Transformer层处理
        # 使用src_key_padding_mask来指定填充位置，而不是mask参数
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=attention_mask)
        
        # 按序列长度降序排序，以便使用pack_padded_sequence
        sorted_seq_lengths, sorted_indices = torch.sort(seq_lengths, descending=True)
        sorted_x = transformer_output[sorted_indices]
        
        # 打包序列
        packed_x = nn.utils.rnn.pack_padded_sequence(
            sorted_x, sorted_seq_lengths.cpu(), batch_first=True
        )
        
        # 前向传播通过Bi-GRU
        packed_output, _ = self.bi_gru(packed_x)
        
        # 解包序列
        output, output_seq_len = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 恢复原始顺序
        _, original_indices = torch.sort(sorted_indices)
        output = output[original_indices]
        
        # 应用输出层
        output = self.output_layer(output)
        
        # 确保output和attention_mask的序列长度匹配
        # 如果output的序列长度小于原始序列长度，进行填充
        if output.size(1) < original_seq_len:
            pad_size = original_seq_len - output.size(1)
            output = torch.cat([output, torch.zeros(output.size(0), pad_size, output.size(2), device=output.device)], dim=1)
        # 如果output的序列长度大于原始序列长度，进行截断
        elif output.size(1) > original_seq_len:
            output = output[:, :original_seq_len, :]
        
        # 计算注意力掩码的反向版本
        attention_mask_rev = (~attention_mask).int()
        
        # 计算加权和
        input_mask_expanded = attention_mask_rev.unsqueeze(-1).expand(output.size()).float()
        sum_embeddings = torch.sum(output * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # 防止除以零
        pooled = sum_embeddings / sum_mask
        
        return pooled


class FeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = None,
        dropout: int = 0.15,
        add_norm: bool = False,
    ) -> None:
        super().__init__()
        if add_norm and (in_features != out_features):
            raise ValueError

        if hidden_features is None:
            hidden_features = 4 * in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu
        self.add_norm = add_norm
        if add_norm:
            self.layernorm = nn.LayerNorm(out_features)

    def _feedforward(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x

    def forward(self, x):
        if self.add_norm:
            return self.layernorm(x + self._feedforward(x))
        return self._feedforward(x)


class CodeChangeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hunk_encoder = AutoModel.from_pretrained("/root/.cache/huggingface/hub/models--microsoft--codebert-base")
        # 移除不再使用的线性层
        # self.hunk_compare_linear1 = nn.Linear(768 * 2, 768)
        # self.hunk_compare_linear2 = nn.Linear(768, 768)
        # 调整输入维度为768（仅减法结果）
        self.hunk_compare_poller = FeedForward(768, 128, 1024)
        self.hunk_reducer = Reducer()
        self.file_reducer = Reducer()
        # 移除不再使用的layernorm层
        # self.layernorm = nn.LayerNorm(768 * 3)

    def encode_hunk(self, input_ids, attention_mask):
        # 确保输入的序列长度不为0
        if input_ids.shape[1] == 0:
            # 如果序列长度为0，返回一个零张量
            return torch.zeros(input_ids.shape[0], 768, device=input_ids.device)
        return self.hunk_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output

    def subtraction(self, added_code, removed_code):
        return added_code - removed_code

    def multiplication(self, added_code, removed_code):
        return added_code * removed_code

    def forward_compare_linear(self, added_code, removed_code):
        concat = torch.cat((removed_code, added_code), dim=1)
        output = self.hunk_compare_linear1(concat)
        output = F.relu(output)
        output = self.hunk_compare_linear2(output)
        return output

    def cosine_similarity(self, added_code, removed_code):
        cosine = nn.CosineSimilarity(eps=1e-6)
        return cosine(added_code, removed_code).view(-1, 1)

    def euclidean_similarity(self, added_code, removed_code):
        euclidean = nn.PairwiseDistance(p=2)
        return euclidean(added_code, removed_code).view(-1, 1)

    def compare_hunk_features(self, hunk_add_feature, hunk_del_feature):
        # 仅保留减法操作，移除乘法和线性变换层
        sub = self.subtraction(hunk_add_feature, hunk_del_feature)  # (batch, 768)
        # 直接使用减法结果作为特征
        out = self.hunk_compare_poller(sub)
        return out

    def forward_hunk(
        self,
        hunk_add_input_ids,
        hunk_add_attention_mask,
        hunk_delete_input_ids,
        hunk_delete_attention_mask,
    ):
        n_batch, n_hunk = hunk_add_input_ids.shape[0], hunk_add_input_ids.shape[1]
        hunk_features = []

        for i in range(n_hunk):
            hunk_add_input_ids_i, hunk_add_attention_mask_i = [], []
            hunk_del_input_ids_i, hunk_del_attention_mask_i = [], []
            for b in range(n_batch):
                hunk_add_input_ids_i.append(hunk_add_input_ids[b][i])
                hunk_add_attention_mask_i.append(hunk_add_attention_mask[b][i])
                hunk_del_input_ids_i.append(hunk_delete_input_ids[b][i])
                hunk_del_attention_mask_i.append(hunk_delete_attention_mask[b][i])

            hunk_add_input_ids_i = torch.stack(hunk_add_input_ids_i, dim=0)
            hunk_add_attention_mask_i = torch.stack(hunk_add_attention_mask_i, dim=0)
            hunk_del_input_ids_i = torch.stack(hunk_del_input_ids_i, dim=0)
            hunk_del_attention_mask_i = torch.stack(hunk_del_attention_mask_i, dim=0)

            # 确保attention_mask是正确的类型（0或1的整数张量）
            hunk_add_attention_mask_i = hunk_add_attention_mask_i.int()
            hunk_del_attention_mask_i = hunk_del_attention_mask_i.int()

            hunk_add_feature_i = self.encode_hunk(
                hunk_add_input_ids_i, hunk_add_attention_mask_i
            )
            hunk_del_feature_i = self.encode_hunk(
                hunk_del_input_ids_i, hunk_del_attention_mask_i
            )
            assert hunk_add_feature_i.shape == hunk_del_feature_i.shape
            assert hunk_add_feature_i.shape == torch.Size([n_batch, 768])

            hunk_feature = self.compare_hunk_features(
                hunk_add_feature_i, hunk_del_feature_i
            )  # (batch, 128)
            hunk_features.append(hunk_feature)

        hunk_features = torch.stack(hunk_features, dim=1)  # (batch, hunk, 128)
        return hunk_features

    def forward(
        self,
        code_add_input_ids,
        code_add_attention_mask,
        code_del_input_ids,
        code_del_attention_mask,
        file_attention_mask,
        hunk_attention_mask,
    ):
        n_batch, n_file, n_hunk = (
            code_add_input_ids.shape[0],
            code_add_input_ids.shape[1],
            code_add_input_ids.shape[2],
        )

        code_features = []
        for i in range(n_file):
            file_add_input_ids_i, file_add_attention_mask_i = [], []
            file_del_input_ids_i, file_del_attention_mask_i = [], []
            for b in range(n_batch):
                file_add_input_ids_i.append(code_add_input_ids[b][i])
                file_add_attention_mask_i.append(code_add_attention_mask[b][i])
                file_del_input_ids_i.append(code_del_input_ids[b][i])
                file_del_attention_mask_i.append(code_del_attention_mask[b][i])

            file_add_input_ids_i = torch.stack(file_add_input_ids_i, dim=0)
            file_add_attention_mask_i = torch.stack(file_add_attention_mask_i, dim=0)
            file_del_input_ids_i = torch.stack(file_del_input_ids_i, dim=0)
            file_del_attention_mask_i = torch.stack(file_del_attention_mask_i, dim=0)

            code_feature = self.forward_hunk(
                file_add_input_ids_i,
                file_add_attention_mask_i,
                file_del_input_ids_i,
                file_del_attention_mask_i,
            )
            code_features.append(code_feature)

        code_features = torch.stack(code_features, dim=1)
        assert code_features.shape == torch.Size([n_batch, n_file, n_hunk, 128])

        files = None
        for f in range(n_file):
            hunks = []
            hunk_attention_masks = []
            for b in range(n_batch):
                hunk_attention_masks.append(hunk_attention_mask[b][f])
            for h in range(n_hunk):
                hunk = []
                for b in range(n_batch):
                    hunk.append(code_features[b][f][h])
                hunk = torch.stack(hunk, dim=0)
                assert hunk.shape == torch.Size([n_batch, 128])
                hunks.append(hunk)
            hunks = torch.stack(hunks, dim=0)
            hunk_attention_masks = torch.stack(hunk_attention_masks, dim=0)
            assert hunks.shape == torch.Size([n_hunk, n_batch, 128])
            assert hunk_attention_masks.shape == torch.Size([n_batch, n_hunk])
            hunks_feature = self.hunk_reducer(hunks, hunk_attention_masks)
            hunks_feature = hunks_feature.unsqueeze(dim=0)

            if files is None:
                files = hunks_feature
            else:
                files = torch.cat((files, hunks_feature), dim=0)
        assert files.shape == torch.Size([n_file, n_batch, 128]), files.shape

        commit_feature = self.file_reducer(files, file_attention_mask)
        assert commit_feature.shape == torch.Size([n_batch, 128])
        return commit_feature


@MODEL_REGISTRY.register("CCModel")
class CCModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.code_change_encoder = CodeChangeEncoder()

        tabular_config = TabularConfig(
            num_labels=10,
            combine_feat_method="gating_on_cat_and_num_feats_then_sum",
            numerical_feat_dim=21,
            numerical_bn=False,
            mlp_dropout=0.15,
        )
        tabular_config.text_feat_dim = 768
        self.feature_combiner = TabularFeatCombiner(tabular_config)
        self.text_code_combiner = FeedForward(768 + 128, 768, 2048)
        self.classifier = nn.Linear(768, 10)

    def forward(
        self,
        input_ids,
        attention_mask,
        codes_add_input_ids,
        codes_add_attention_mask,
        codes_delete_input_ids,
        codes_delete_attention_mask,
        file_attention_mask,
        hunk_attention_mask,
        numerical_features,
        **kwargs,
    ):
        msg_embeding = self.code_change_encoder.hunk_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output

        code_embeding = self.code_change_encoder(
            code_add_input_ids=codes_add_input_ids,
            code_add_attention_mask=codes_add_attention_mask,
            code_del_input_ids=codes_delete_input_ids,
            code_del_attention_mask=codes_delete_attention_mask,
            file_attention_mask=file_attention_mask,
            hunk_attention_mask=hunk_attention_mask,
        )

        combined = self.text_code_combiner(
            torch.cat((msg_embeding, code_embeding), dim=1)
        )

        commit_embeding = self.feature_combiner(
            combined,
            numerical_feats=numerical_features,
        )
        return self.classifier(commit_embeding)


@MODEL_REGISTRY.register("CodeBERTBaseline")
class CCModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.roberta = AutoModel.from_pretrained("roberta-base")
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.classifier = FeedForward(768 * 2, 10, 768 * 4)

    def forward(
        self,
        input_ids,
        attention_mask,
        diff_input_ids,
        diff_attention_mask,
        **kwargs,
    ):
        msg_embeding = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output
        diff_embedding = self.codebert(
            input_ids=diff_input_ids, attention_mask=diff_attention_mask
        ).pooler_output
        return self.classifier(torch.cat((msg_embeding, diff_embedding), dim=1))


@MODEL_REGISTRY.register("CodeFeatModel")
class CodeFeatModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.code_change_encoder = CodeChangeEncoder()
        tabular_config = TabularConfig(
            num_labels=10,
            combine_feat_method="gating_on_cat_and_num_feats_then_sum",
            numerical_feat_dim=21,
            numerical_bn=False,
            mlp_dropout=0.15,
        )
        tabular_config.text_feat_dim = 128
        self.feature_combiner = TabularFeatCombiner(tabular_config)
        self.classifier = nn.Linear(128, 10)

    def forward(
        self,
        codes_add_input_ids,
        codes_add_attention_mask,
        codes_delete_input_ids,
        codes_delete_attention_mask,
        file_attention_mask,
        hunk_attention_mask,
        numerical_features,
        **kwargs,
    ):
        code_embeding = self.code_change_encoder(
            code_add_input_ids=codes_add_input_ids,
            code_add_attention_mask=codes_add_attention_mask,
            code_del_input_ids=codes_delete_input_ids,
            code_del_attention_mask=codes_delete_attention_mask,
            file_attention_mask=file_attention_mask,
            hunk_attention_mask=hunk_attention_mask,
        )

        commit_embeding = self.feature_combiner(
            code_embeding,
            numerical_feats=numerical_features,
        )
        return self.classifier(commit_embeding)


@MODEL_REGISTRY.register("MessageCodeModel")
class MessageModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.code_change_encoder = CodeChangeEncoder()
        self.text_code_combiner = FeedForward(768 + 128, 768, 2048)
        self.classifier = nn.Linear(768, 10)

    def forward(
        self,
        input_ids,
        attention_mask,
        codes_add_input_ids,
        codes_add_attention_mask,
        codes_delete_input_ids,
        codes_delete_attention_mask,
        file_attention_mask,
        hunk_attention_mask,
        **kwargs,
    ):
        msg_embeding = self.code_change_encoder.hunk_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output
        code_embeding = self.code_change_encoder(
            code_add_input_ids=codes_add_input_ids,
            code_add_attention_mask=codes_add_attention_mask,
            code_del_input_ids=codes_delete_input_ids,
            code_del_attention_mask=codes_delete_attention_mask,
            file_attention_mask=file_attention_mask,
            hunk_attention_mask=hunk_attention_mask,
        )

        combined = self.text_code_combiner(
            torch.cat((msg_embeding, code_embeding), dim=1)
        )

        return self.classifier(combined)


@MODEL_REGISTRY.register("MessageFeatModel")
class MessageFeatModel(nn.Module):
    def __init__(self, config=None) -> None:
        super().__init__()
        self.config = config
        self.message_encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Linear(768, 10)
        tabular_config = TabularConfig(
            num_labels=10,
            combine_feat_method="gating_on_cat_and_num_feats_then_sum",
            numerical_feat_dim=29,
            numerical_bn=False,
            mlp_dropout=0.15,
        )
        tabular_config.text_feat_dim = 768
        self.feature_combiner = TabularFeatCombiner(tabular_config)

    def forward(
        self,
        input_ids,
        attention_mask,
        numerical_features,
        **kwargs,
    ):
        msg_embeding = self.message_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output
        commit_embeding = self.feature_combiner(
            msg_embeding,
            numerical_feats=numerical_features,
        )
        out = self.classifier(commit_embeding)
        return out
