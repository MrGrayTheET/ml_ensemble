import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.0, use_time_distributed=True, additional_context=False, return_gate=False):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size if output_size is not None else input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_time_distributed = use_time_distributed
        self.additional_context = additional_context
        self.return_gate = return_gate

        if self.output_size != self.input_size:
            self.skip_layer = nn.Linear(self.input_size, self.output_size)
        else:
            self.skip_layer = None

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        if self.additional_context:
            self.context_fc = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.elu = nn.ELU()
        self.gate_fc = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0 else None
        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(self, x, context=None):
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x

        hidden = self.fc1(x)

        if self.additional_context and context is not None:
            hidden = hidden + self.context_fc(context)

        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)

        if self.dropout_layer is not None:
            hidden = self.dropout_layer(hidden)

        gate = self.sigmoid(self.gate_fc(hidden))
        gated_hidden = hidden * gate

        output = self.layer_norm(skip + gated_hidden)

        if self.return_gate:
            return output, gate
        else:
            return output

class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout: float = 0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = self.d_v = d_model // n_head
        self.dropout = dropout

        self.qs_layers = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)])
        self.ks_layers = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)])
        self.vs_layer = nn.Linear(d_model, self.d_v, bias=False)
        self.vs_layers = nn.ModuleList([self.vs_layer for _ in range(n_head)])

        self.attention = ScaledDotProductAttention(attn_dropout=dropout)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        heads = []
        attns = []
        for i in range(self.n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            head, attn = self.attention(qs, ks, vs, mask)
            head = self.dropout_layer(head)
            heads.append(head)
            attns.append(attn)

        if self.n_head > 1:
            head = torch.stack(heads, dim=0)
            attn = torch.stack(attns, dim=0)
            outputs = torch.mean(head, dim=0)
        else:
            outputs = heads[0]
            attn = attns[0]

        outputs = self.w_o(outputs)
        outputs = self.dropout_layer(outputs)
        return outputs, attn


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.0, use_time_distributed=True,
                 additional_context=False, return_gate=False):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size if output_size is not None else input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_time_distributed = use_time_distributed
        self.additional_context = additional_context
        self.return_gate = return_gate

        if self.output_size != self.input_size:
            self.skip_layer = nn.Linear(self.input_size, self.output_size)
        else:
            self.skip_layer = None

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        if self.additional_context:
            self.context_fc = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.elu = nn.ELU()
        self.gate_fc = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0 else None
        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(self, x, context=None):
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x

        hidden = self.fc1(x)
        if self.additional_context and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)

        if self.dropout_layer is not None:
            hidden = self.dropout_layer(hidden)

        gate = self.sigmoid(self.gate_fc(hidden))
        gated_hidden = hidden * gate

        output = self.layer_norm(skip + gated_hidden)

        if self.return_gate:
            return output, gate
        else:
            return output


class TftDeepMomentumNetworkModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size, num_heads, dropout_rate, num_encoder_steps,
                 category_counts, static_input_loc, known_regular_inputs, known_categorical_inputs,
                 force_output_sharpe_length=None):
        super(TftDeepMomentumNetworkModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_encoder_steps = num_encoder_steps
        self.category_counts = category_counts
        self.static_input_loc = static_input_loc
        self.known_regular_inputs = known_regular_inputs
        self.known_categorical_inputs = known_categorical_inputs
        self.force_output_sharpe_length = force_output_sharpe_length

        self.num_categorical_variables = len(self.category_counts)
        self.num_regular_variables = self.input_size - self.num_categorical_variables

        self.embeddings = nn.ModuleList(
            [nn.Embedding(category_size, self.hidden_layer_size) for category_size in self.category_counts])

        self.static_context_variable_selection = GatedResidualNetwork(
            self.hidden_layer_size, self.hidden_layer_size, dropout=self.dropout_rate, use_time_distributed=False
        )
        self.static_context_enrichment = GatedResidualNetwork(
            self.hidden_layer_size, self.hidden_layer_size, dropout=self.dropout_rate, use_time_distributed=False
        )
        self.static_context_state_h = GatedResidualNetwork(
            self.hidden_layer_size, self.hidden_layer_size, dropout=self.dropout_rate, use_time_distributed=False
        )
        self.static_context_state_c = GatedResidualNetwork(
            self.hidden_layer_size, self.hidden_layer_size, dropout=self.dropout_rate, use_time_distributed=False
        )

        self.lstm = nn.LSTM(
            input_size=self.hidden_layer_size,
            hidden_size=self.hidden_layer_size,
            batch_first=True,
            dropout=0
        )

        self.self_attn_layer = InterpretableMultiHeadAttention(
            n_head=self.num_heads, d_model=self.hidden_layer_size, dropout=self.dropout_rate
        )

        self.temporal_feature_layer = GatedResidualNetwork(
            input_size=self.hidden_layer_size, hidden_size=self.hidden_layer_size, dropout=self.dropout_rate,
            use_time_distributed=True
        )

        self.static_enrichment_grn = GatedResidualNetwork(
            input_size=self.hidden_layer_size, hidden_size=self.hidden_layer_size, dropout=self.dropout_rate,
            use_time_distributed=True, additional_context=True
        )

        self.decoder_grn = GatedResidualNetwork(
            input_size=self.hidden_layer_size, hidden_size=self.hidden_layer_size, dropout=self.dropout_rate,
            use_time_distributed=True
        )

        self.output_layer = nn.Linear(self.hidden_layer_size, self.output_size)
        self.output_tanh = nn.Tanh()

    def forward(self, x):
        # Transform inputs
        known_inputs, unknown_inputs, static_ctx, static_weights = self.transform_inputs(x)

        # Combine inputs
        if unknown_inputs is not None and known_inputs is not None:
            historical_inputs = torch.cat([unknown_inputs, known_inputs], dim=-1)
        elif unknown_inputs is not None:
            historical_inputs = unknown_inputs
        else:
            historical_inputs = known_inputs

        # Get static context for variable selection
        static_context_variable_selection = self.static_context_variable_selection(static_ctx)
        static_context_enrichment = self.static_context_enrichment(static_ctx)
        static_context_state_h = self.static_context_state_h(static_ctx)
        static_context_state_c = self.static_context_state_c(static_ctx)

        # LSTM layer
        h0 = static_context_state_h.unsqueeze(0).repeat(1, x.size(0), 1)
        c0 = static_context_state_c.unsqueeze(0).repeat(1, x.size(0), 1)
        lstm_output, _ = self.lstm(historical_inputs, (h0, c0))

        # Temporal feature layer
        temporal_features = self.temporal_feature_layer(lstm_output) + historical_inputs

        # Static enrichment
        static_context_expanded = static_context_enrichment.unsqueeze(1).expand(-1, temporal_features.size(1), -1)
        enriched = self.static_enrichment_grn(temporal_features, static_context_expanded)

        # Decoder self-attention
        mask = self.get_decoder_mask(enriched.size(1)).to(enriched.device)
        attn_output, self_attn_weights = self.self_attn_layer(enriched, enriched, enriched, mask=mask)
        attn_output = attn_output + enriched

        # Nonlinear processing on outputs
        decoder_output = self.decoder_grn(attn_output)

        # Final skip connection
        transformer_output = decoder_output + temporal_features

        # Output layer
        if self.force_output_sharpe_length:
            output_tensor = transformer_output[:, -self.force_output_sharpe_length:, :]
        else:
            output_tensor = transformer_output

        outputs = self.output_tanh(self.output_layer(output_tensor))

        # Save attention components for interpretability
        self.attention_components = {
            "decoder_self_attn": self_attn_weights,
            "static_flags": static_weights[:, :, 0] if static_weights is not None else None,
        }

        return outputs

    def get_decoder_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return ~mask

    def transform_inputs(self, x):
        # Split regular and categorical inputs
        regular_inputs = x[:, :, :self.num_regular_variables]
        categorical_inputs = x[:, :, self.num_regular_variables:].long()

        # Transform regular inputs
        transformed_regular = []
        for i in range(self.num_regular_variables):
            regular_var = regular_inputs[:, :, i:i + 1].contiguous()
            transformed = nn.Linear(1, self.hidden_layer_size)(regular_var)
            transformed_regular.append(transformed)

        # Transform categorical inputs
        transformed_categorical = []
        for i in range(self.num_categorical_variables):
            cat_var = categorical_inputs[:, :, i]
            embedded = self.embeddings[i](cat_var)
            transformed_categorical.append(embedded)

        # Process static inputs
        if self.static_input_loc:
            static_regular = []
            for i in range(self.num_regular_variables):
                if i in self.static_input_loc:
                    static_regular.append(transformed_regular[i][:, 0, :].unsqueeze(1))

            static_categorical = []
            for i in range(self.num_categorical_variables):
                if i + self.num_regular_variables in self.static_input_loc:
                    static_categorical.append(transformed_categorical[i][:, 0, :].unsqueeze(1))

            static_inputs = torch.cat(static_regular + static_categorical, dim=1)
            static_ctx, static_weights = self.process_static_inputs(static_inputs)
        else:
            static_ctx = None
            static_weights = None

        # Group known inputs
        known_regular = []
        for i in range(self.num_regular_variables):
            if i in self.known_regular_inputs and i not in self.static_input_loc:
                known_regular.append(transformed_regular[i])

        known_categorical = []
        for i in range(self.num_categorical_variables):
            if (i in self.known_categorical_inputs and
                    i + self.num_regular_variables not in self.static_input_loc):
                known_categorical.append(transformed_categorical[i])

        # Group unknown inputs
        unknown_regular = []
        for i in range(self.num_regular_variables):
            if i not in self.known_regular_inputs and i not in self.static_input_loc:
                unknown_regular.append(transformed_regular[i])

        unknown_categorical = []
        for i in range(self.num_categorical_variables):
            if (i not in self.known_categorical_inputs and
                    i + self.num_regular_variables not in self.static_input_loc):
                unknown_categorical.append(transformed_categorical[i])

        # Combine inputs
        if known_regular or known_categorical:
            known_combined = torch.cat(known_regular + known_categorical, dim=-1)
        else:
            known_combined = None

        if unknown_regular or unknown_categorical:
            unknown_combined = torch.cat(unknown_regular + unknown_categorical, dim=-1)
        else:
            unknown_combined = None

        return known_combined, unknown_combined, static_ctx, static_weights

    def process_static_inputs(self, static_inputs):
        batch_size = static_inputs.size(0)
        num_static = static_inputs.size(1)

        flatten = static_inputs.view(batch_size, -1)

        static_weights = self.static_selection(flatten)
        static_weights = F.softmax(static_weights, dim=-1).unsqueeze(-1)

        transformed_static = []
        for i in range(num_static):
            static_var = static_inputs[:, i:i + 1, :]
            transformed = self.static_grns[i](static_var)
            transformed_static.append(transformed)

        transformed_static = torch.cat(transformed_static, dim=1)

        static_ctx = torch.sum(static_weights * transformed_static, dim=1)

        return static_ctx, static_weights