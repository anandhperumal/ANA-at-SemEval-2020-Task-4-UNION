from transformers.modeling_gpt2 import *

class MTDNN(GPT2DoubleHeadsModel):
    def __init__(self, config):
        super(MTDNN, self).__init__(config)
        config.num_labels = 3
        self.multiple_choice_head_commonsense = SequenceSummary(config)

        config.num_labels = 4
        self.multiple_choice_head_openbook = SequenceSummary(config)
        #
        config.num_labels = 5
        self.multiple_choice_head_cose = SequenceSummary(config)


    def forward(self, input_ids=None, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                mc_token_ids=None, lm_labels=None, mc_labels=None, flag=True, task = 0):
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               )

        hidden_states = transformer_outputs[0]
        if flag:
            lm_logits = self.lm_head(hidden_states)

        # 50265 - openbook
        #commonsensetask 50263
        # COSE - 64
        # task = input_ids[0][0][1]
        if task == 50263:
            mc_logits = self.multiple_choice_head_commonsense(hidden_states, mc_token_ids).squeeze(-1)

        elif task == 50265:
            mc_logits = self.multiple_choice_head_openbook(hidden_states, mc_token_ids).squeeze(-1)

        elif task == 50264:
            mc_logits = self.multiple_choice_head_cose(hidden_states, mc_token_ids).squeeze(-1)
        elif task == 0:
            mc_logits = ""
        else:
            import sys
            sys.exit("Error in MTDNN")

        outputs = (mc_logits,) + transformer_outputs[1:]
        if flag:
            outputs = (lm_logits, mc_logits) + transformer_outputs[1:]

        if mc_labels is not None:
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)),
                            mc_labels.view(-1))
            outputs = (loss,) + outputs
        if lm_labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
