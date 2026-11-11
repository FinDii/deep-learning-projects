import torch
from torch import Tensor, nn

from .attention import AdditiveAttention


class RNNDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        vocab_size: int,
        feature_dim: int,
        dropout=0.5,
        rnn_cell=nn.LSTMCell,
    ):
        """Initialize RNNDecoder.

        Args:
            embed_dim (int): Embedding dimension.
            hidden_dim (int): Hidden dimension.
            vocab_size (int): Vocabulary size.
            feature_dim (int): Feature dimension.
            dropout (float): Dropout rate. Default: 0.5.
            rnn_cell (Module): RNN cell module. Default: nn.LSTMCell.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # ============ TODO: Init layers ============
        # Decoder expects these layers:
        # - embedding: input embedding 输入词嵌入层（将词索引转为稠密向量）
        # - lstm: lstm cell
        # - init_h: linear projection to init hidden state from image features
        #    从图像特征投影到 LSTM 的初始隐藏状态 h0
        # - init_c: linear projection to init cell state from image features
        #    从图像特征投影到 LSTM 的初始细胞状态 c0
        # - dropout: output dropout layer
        # - fc: output linear projection全连接层
        # raise NotImplementedError
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = rnn_cell(embed_dim, hidden_dim)
        self.lstm_cell = self.lstm
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # ===========================================

        # initialize some layers with the uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Initialize hidden states for RNNDecoder.

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).

        Returns:
            tuple: (hidden_state, cell_state) both with shape (batch, hidden_dim).
        """
        # ========== TODO: Init hidden states ==========
        # raise NotImplementedError
        pooled_features = features.mean(dim=1)# 对像素维度取平均，使得 features 变为 (batch, feature_dim)
        h = self.init_h(pooled_features)   # (batch_size, hidden_size)
        c = self.init_c(pooled_features)
        # ==============================================
        return h, c

    def forward(self, features: Tensor, captions: Tensor):
        """Forward pass of RNNDecoder.

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).
            captions (Tensor): Input captions with shape (batch, max_caption_length).

        Returns:
            Tensor: Output predictions with shape (batch, max_caption_length, vocab_size).
        """
        B = features.shape[0]
        T = captions.shape[1]

        # 词嵌入
        embeddings: Tensor = self.embedding(captions)
        # 初始化隐藏状态
        h, c = self.init_hidden_state(features)

        outputs: list[Tensor] = []
        # Decode step by step
        for t in range(T - 1):
            # ======== TODO: implement decode step ========
            # Hint 1: apply dropout just before fc
            # Hint 2: use teacher-forcing here
            # raise NotImplementedError
            current_embed = embeddings[:, t, :] # 取当前时间步的词嵌入 (batch_size, embed_size)
            h, c = self.lstm_cell(current_embed, (h, c))   # 传入 LSTM Cell，更新 h 和 c
            h_dropped = self.dropout(h)   # 经过 Dropout
            
            logits = self.fc(h_dropped)
            outputs.append(logits)   # 堆叠所有时间步的输出，调整维度为 (batch_size, seq_len, vocab_size)
            # =============================================

        # 堆叠输出
        return torch.stack(outputs, dim=1)

    def generate_caption(
        self, features: Tensor, word_map: dict[str, int], max_len=30, beam_size=1
    ):
        """Generate captions with greedy decoding or beam search (during test time).

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).
            word_map (dict[str, int]): Word map from token to index.
            max_len (int): Maximum caption length. Default: 30.
            beam_size (int): Beam size for beam search. Default: 1 (greedy decoding).

        Returns:
            list[list[int]]: Generated captions.
        """
        B = features.shape[0]
        features = features.view(B, -1, features.shape[-1])
        device = features.device

        # 获取特殊标记
        start_token = word_map["<start>"]
        end_token = word_map["<end>"]

        if beam_size == 1:
            # 使用 greedy decoding
            # 初始化隐藏状态
            h, c = self.init_hidden_state(features)

            input_tokens = torch.full(
                (B,),
                start_token,
                dtype=torch.long,
                device=device
            )

            captions = [[start_token] for _ in range(B)]
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            # 逐词生成
            for _ in range(max_len - 1):
                # ======== TODO: implement generate step ========
                # Hint 1: apply dropout just before fc
                # Hint 2: a caption ends when it reaches the end token
                # 1. 获取当前 predictions
                # 2. 更新 captions
                # raise NotImplementedError
                current_embed = self.embedding(input_tokens) # 取当前时间步的词嵌入 (batch_size, embed_size)
                h, c = self.lstm_cell(current_embed, (h, c))   # 传入 LSTM Cell，更新 h 和 c  
                logits = self.fc(self.dropout(h))  # 经过 Dropout
                next_tokens = logits.argmax(dim=1)  # 贪心选择概率最高的词

                for i, token in enumerate(next_tokens.tolist()):
                    if not finished[i]:
                        captions[i].append(token)

                finished = finished | (next_tokens == end_token)
                input_tokens = next_tokens

                if finished.all():
                    break
            for i in range(B):
                if captions[i][-1] != end_token:
                    captions[i].append(end_token)
                
                # =============================================
            return captions
            
        else:
            # 使用 beam search
            captions: list[list[int]] = []
            # 对每个图像单独进行 beam search
            for i in range(B):
                # 获取单个图像的特征
                img_features = features[i : i + 1]
                # 初始化隐藏状态
                h, c = self.init_hidden_state(img_features)

                # ======== Optional TODO: implement beam search ========
                # Hint: you may need to store beam as (log_prob, hidden_state, cell_state, tokens)
                # 1. Initialize beam with start token
                # 2. Repeat for max_len - 1 steps:
                # 2.1. Compute log probabilities for all possible next tokens
                #      Note: compute only for non-terminated sequences
                # 2.2. Update beam with top-k (k=beam_size) candidates
                # 2.3. Sort beam by log probability, and keep top-k beams
                # 2.4. Stop if all sequences end with end token
                # 3. Select the sequence with highest log probability
                # raise NotImplementedError

                # beam 存储为 (log_prob, hidden, cell, sequence)
                beam = [(0.0, h, c, [])]
                
                for step in range(max_len - 1):
                    new_beam = []
                    # 对每个 beam 进行拓展
                    for log_prob, h_beam, c_beam, seq in beam:
                        # 如果当前序列已经以 <end> 结束，则直接保留
                        if seq and seq[-1] == end_token:
                            new_beam.append((log_prob, h_beam, c_beam, seq))
                            continue

                        # 当前输入 token
                        current_token = start_token if len(seq) == 0 else seq[-1]
                        current_tensor = torch.tensor([current_token], device=device).long()  # (1,)
                        embed = self.embedding(current_tensor)                                 # (1, embed_dim)
                
                        # 更新LSTM
                        h_new, c_new = self.lstm_cell(embed, (h_beam, c_beam)) 

                        # 输出logits
                        logits = self.fc(self.dropout(h_new))
                        log_probs = torch.log_softmax(logits, dim=1)

                        # 取 top-k（k = beam_size）个候选
                        top_log_probs, top_indices = log_probs.topk(beam_size, dim=1)

                        for k in range(beam_size):
                            cand_log_prob = log_prob + top_log_probs[0, k].item()
                            cand_token = top_indices[0, k].item()
                            new_seq = seq + [cand_token]
                            new_beam.append((cand_log_prob, h_new, c_new, new_seq))
                   
                    # 按 log_prob 降序排序，保留 beam_size 个候选
                    new_beam.sort(key=lambda x: x[0], reverse=True)
                    beam = new_beam[:beam_size]
                
                    # 如果所有 beam 都以 <end> 结尾，提前结束
                    if all(seq and seq[-1] == end_token for _, _, _, seq in beam):
                        break

                # 选择概率最高的序列
                best_seq = beam[0][3]
                full_seq = [start_token] + best_seq # 将 <start> 添加到开头
                
                # 若未以 <end> 结尾，添加（通常 beam 会在结束前停止，但为安全）
                if full_seq[-1] != end_token:
                    full_seq.append(end_token)
                    
                captions.append(full_seq)
                # ======================================================

            return captions


class RNNDecoderWithAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        vocab_size: int,
        feature_dim: int,
        dropout=0.5,
        rnn_cell=nn.LSTMCell,
        attn_impl=AdditiveAttention,
    ):
        """Initialize RNNDecoder.

        Args:
            embed_dim (int): Embedding dimension.
            hidden_dim (int): Hidden dimension.
            vocab_size (int): Vocabulary size.
            feature_dim (int): Feature dimension.
            dropout (float): Dropout rate. Default: 0.5.
            rnn_cell (Module): RNN cell module. Default: nn.LSTMCell.
            attn_impl (Module): Attention module. Default: AdditiveAttention.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        # ============ TODO: Init layers ============
        # Decoder expects these layers:
        # - embedding: input embedding
        # - lstm: lstm cell
        # - attention: attention module
        # - init_h: linear projection to init hidden state from image features
        # - init_c: linear projection to init cell state from image features
        # - dropout: output dropout layer
        # - fc: output linear projection
        # raise NotImplementedError

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = rnn_cell(embed_dim + feature_dim, hidden_dim)
        self.lstm_cell = self.lstm
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # 加入注意力相关层
        self.attention = AdditiveAttention(feature_dim, hidden_dim)
        
        # ===========================================

        # initialize some layers with the uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Initialize hidden states for RNNDecoder.

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).

        Returns:
            tuple: (hidden_state, cell_state) both with shape (batch, hidden_dim).
        """
        # ========== TODO: Init hidden states ==========
        # Same as RNNDecoder, you can copy the code
        # raise NotImplementedError
        pooled_features = features.mean(dim=1)# 对像素维度取平均，使得 features 变为 (batch, feature_dim)
        h = self.init_h(pooled_features)   # (batch_size, hidden_size)
        c = self.init_c(pooled_features)
        # ==============================================
        return h, c

    def forward(self, features: Tensor, captions: Tensor):
        """Forward pass of RNNDecoder.

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).
            captions (Tensor): Input captions with shape (batch, max_caption_length).

        Returns:
            Tensor: Output predictions with shape (batch, max_caption_length, vocab_size).
        """
        B = features.shape[0]
        T = captions.shape[1]

        # 词嵌入
        embeddings = self.embedding(captions)
        # 初始化隐藏状态
        h, c = self.init_hidden_state(features)

        outputs = []
        # 逐词解码
        for t in range(T - 1):
            # ======== TODO: implement decode step ========
            # Hint 1: apply dropout just before fc
            # Hint 2: use teacher-forcing here
            # raise NotImplementedError
            current_embed = embeddings[:, t, :] # 取当前时间步的词嵌入 (batch_size, embed_size)
            # attention: 用当前 hidden state 查询图像区域特征
            context, alpha = self.attention(h, features)

            # 将 context 投影到 embed_dim 后与词嵌入相加
            context, alpha = self.attention(h, features)
            lstm_input = torch.cat([current_embed, context], dim=1)

            h, c = self.lstm_cell(lstm_input, (h, c))
            logits = self.fc(self.dropout(h))
            outputs.append(logits)
            # =============================================

        # 堆叠输出
        return torch.stack(outputs, dim=1)

    def generate_caption(
        self, features: Tensor, word_map: dict[str, int], max_len=30, beam_size=1
    ):
        """Generate captions with greedy decoding or beam search (during test time).

        Args:
            features (Tensor): Input features with shape (batch, num_pixels, feature_dim).
            word_map (dict[str, int]): Word map from token to index.
            max_len (int): Maximum caption length. Default: 30.
            beam_size (int): Beam size for beam search. Default: 1 (greedy decoding).

        Returns:
            list[list[int]]: Generated captions.
        """
        B = features.shape[0]
        features = features.view(B, -1, features.shape[-1])
        device = features.device

        # 获取特殊标记
        start_token = word_map["<start>"]
        end_token = word_map["<end>"]

        if beam_size == 1:
            # 使用 greedy decoding
            # 初始化隐藏状态
            h, c = self.init_hidden_state(features)

            input_tokens = torch.full(
                (B,),
                start_token,
                dtype=torch.long,
                device=device
            )

            captions = [[start_token] for _ in range(B)]
            finished = torch.zeros(B, dtype=torch.bool, device=device)

            # 逐词生成
            for _ in range(max_len - 1):
                # ======== TODO: implement generate step ========
                # Hint 1: apply dropout just before fc
                # Hint 2: a caption ends when it reaches the end token
                # 1. 获取当前 predictions
                # 2. 更新 captions
                # raise NotImplementedError
                current_embed = self.embedding(input_tokens) 

                # attention
                context, alpha = self.attention(h, features)
                lstm_input = torch.cat([current_embed, context], dim=1)

                h, c = self.lstm_cell(lstm_input, (h, c))   # 更新 h 和 c  
                logits = self.fc(self.dropout(h))  # 经过 Dropout

                next_tokens = logits.argmax(dim=1)  # 贪心选择概率最高的词

                for i, token in enumerate(next_tokens.tolist()):
                    if not finished[i]:
                        captions[i].append(token)

                finished = finished | (next_tokens == end_token)
                input_tokens = next_tokens

                if finished.all():
                    break
                    
            for i in range(B):
                if captions[i][-1] != end_token:
                    captions[i].append(end_token)
                    
                # =============================================
            return captions

            
        else:
            # 使用 beam search
            captions: list[list[int]] = []
            # 对每个图像单独进行 beam search
            for i in range(B):
                # 获取单个图像的特征
                img_features = features[i : i + 1]
                # 初始化隐藏状态
                h, c = self.init_hidden_state(img_features)

                # ======== Optional TODO: implement beam search ========
                # Hint: you may need to store beam as (log_prob, hidden_state, cell_state, tokens)
                # 1. Initialize beam with start token
                # 2. Repeat for max_len - 1 steps:
                # 2.1. Compute log probabilities for all possible next tokens
                #      Note: compute only for non-terminated sequences
                # 2.2. Update beam with top-k (k=beam_size) candidates
                # 2.3. Sort beam by log probability, and keep top-k beams
                # 2.4. Stop if all sequences end with end token
                # 3. Select the sequence with highest log probability
                # raise NotImplementedError

                # beam 存储为 (log_prob, hidden, cell, sequence)
                beam = [(0.0, h, c, [])]
                
                for step in range(max_len - 1):
                    new_beam = []
                    # 对每个 beam 进行拓展
                    for log_prob, h_beam, c_beam, seq in beam:
                        # 如果当前序列已经以 <end> 结束，则直接保留
                        if seq and seq[-1] == end_token:
                            new_beam.append((log_prob, h_beam, c_beam, seq))
                            continue

                        current_token = start_token if len(seq) == 0 else seq[-1]
                        current_tensor = torch.tensor([current_token], device=device).long()
                        embed = self.embedding(current_tensor)

                        # 注意力机制
                        context, _ = self.attention(h_beam, img_features)
                        lstm_input = torch.cat([embed, context], dim=1)
                
                        # 更新LSTM
                        h_new, c_new = self.lstm_cell(lstm_input, (h_beam, c_beam)) 

                        # 输出logits
                        logits = self.fc(self.dropout(h_new))

                        log_probs = torch.log_softmax(logits, dim=1)

                        # 取 top-k（k = beam_size）个候选
                        top_log_probs, top_indices = log_probs.topk(beam_size, dim=1)

                        for k in range(beam_size):
                            cand_log_prob = log_prob + top_log_probs[0, k].item()
                            cand_token = top_indices[0, k].item()
                            new_seq = seq + [cand_token]
                            new_beam.append((cand_log_prob, h_new, c_new, new_seq))
        
                    # 排序并保留 top‑beam_size
                    new_beam.sort(key=lambda x: x[0], reverse=True)
                    beam = new_beam[:beam_size]

                    # 如果所有 beam 都以 <end> 结尾，提前结束
                    if all(seq and seq[-1] == end_token for _, _, _, seq in beam):
                        break

                # 选择概率最高的序列
                best_seq = beam[0][3]
                full_seq = [start_token] + best_seq # 将 <start> 添加到开头
                
                # 若未以 <end> 结尾，添加（通常 beam 会在结束前停止，但为安全）
                if full_seq[-1] != end_token:
                    full_seq.append(end_token)
                captions.append(full_seq)
                # ======================================================

            return captions
