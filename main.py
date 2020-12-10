import logging
import torch.nn.functional as F
from flask import Flask, request
import torch
from transformers import GPT2Tokenizer
app = Flask(__name__)

class CommonSenseResponseGeneration():
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if CommonSenseResponseGeneration.__instance == None:
            CommonSenseResponseGeneration()
        return CommonSenseResponseGeneration.__instance

    def __init__(self):
        self.tokenizer = None
        self.min_length = 1
        self.max_length = 128
        self.max_text = 20
        self.device = "cpu"
        self.temperature = 0.9
        self.top_k = 0
        self.top_p = 0.9
        self.no_sample = False
        self.TOKENIZER_DIR = 'model/'
        self.MODEL_DIR = 'model/commonsense.pt'
        self.SPECIAL_TOKENS = ['<pad>', '<eos>', '<rstokn>', '<bos>', '<question>', '<commonsensetask>', '<CoSE>',
                               '<openBook>']
        self.model = torch.load(self.MODEL_DIR, map_location='cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.TOKENIZER_DIR, do_lower_case=True)

        """ Virtually private constructor. """
        if CommonSenseResponseGeneration.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            CommonSenseResponseGeneration.__instance = self

    def top_filtering(self, logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
                top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                    whose total probability mass is greater than or equal to the threshold top_p.
                    In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                    the threshold top_p.
                threshold: a minimal threshold to keep logits
        """
        assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Back to unsorted indices and set them to -infinity
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value

        return logits

    def building_input(self, tokenizer, falsesent, pad, current_output):

        pad = [tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS[0])]
        eos = [tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS[1])]
        rstokn = [tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS[2])]
        bos = [tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS[3])]
        questons = [tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS[4])]
        commonsensetask = [tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS[5])]
        COSE = [tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS[6])]
        openBook = [tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS[7])]

        input_id = bos + commonsensetask + rstokn + falsesent + rstokn + current_output
        # input_id = bos + falsesent + rstokn + current_output

        # input_id = torch.tensor(input_id).type(torch.cuda.LongTensor)
        input_id = torch.tensor(input_id).type(torch.LongTensor)

        tt_id = torch.tensor((len(commonsensetask) + 1) * rstokn + (len(falsesent) + 1) * questons + (
                len(current_output) + 1) * rstokn).type(torch.LongTensor)
        # tt_id = torch.tensor((len(falsesent) + 1) * questons + (len(current_output) + 1) * rstokn).type(torch.cuda.LongTensor)
        return input_id, tt_id

    def sample_sequence(self, inputs, tokenizer, model, pad, flag=False):
        special_tokens_ids = tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS)
        # input_ids = building_input(tokenizer,input_ids,pad)
        model.eval()
        model.to(self.device)
        current_output = []
        for i in range(self.max_text):
            # input_ids = [inputs[0] + current_output] if current_output != [] else inputs
            input_ids, tt_id = self.building_input(tokenizer, inputs, pad, current_output)
            input_ids, tt_id = input_ids.unsqueeze(0), tt_id.unsqueeze(0)
            # with torch.no_grad:
            logits = model(input_ids.to(self.device), token_type_ids=tt_id.to(self.device), task=50263)
            if isinstance(logits, tuple):  # for gpt2 and maybe others
                logits = logits[0]
            logits = logits[0, -1, :] / self.temperature
            logits = self.top_filtering(logits, top_k=self.top_k, top_p=self.top_p)
            probs = F.softmax(logits, dim=-1)

            prev = torch.topk(probs, 1)[1] if self.no_sample else torch.multinomial(probs, 1)

            if flag and prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())

        return current_output

    def run(self, raw_text):
        pad = [self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS[0])]
        falsesent = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(raw_text))
        output = self.sample_sequence(falsesent, self.tokenizer, self.model, pad, True)
        out_text = self.tokenizer.decode(output, skip_special_tokens=True)
        return out_text


csrg = CommonSenseResponseGeneration.getInstance()
@app.route('/', methods=["POST"])
def hello():
    response = csrg.run(request.get_json()['text'])
    return response

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    # sudo docker build -t common_sense:0.0.1 .
    # sudo docker run -d -p 8080:8080 common_sense:0.0.1
    app.run(host='0.0.0.0', port=8080)
