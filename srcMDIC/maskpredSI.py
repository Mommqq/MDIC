import torch
import torch.nn as nn
import torch.nn.functional as F
import re


class MaskedObjectPredictor(nn.Module):
    def __init__(self, cliptokenizer, cliptextmodel, vocab=None, device="cuda", nhead=4, num_layers=2):
        super().__init__()
        self.device = device
        self.vocab = vocab
        self.num_classes = len(vocab)


        self.tokenizer = cliptokenizer
        self.text_encoder = cliptextmodel
        special_tokens_dict = {"additional_special_tokens": ["[MASK]"]}
        num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
        if num_added > 0:
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        text_hidden = self.text_encoder.config.hidden_size
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(320, 512),
            nn.ReLU()
        )
        # mask_hidden = 128
        mask_hidden = 512
        fusion_dim = text_hidden + mask_hidden
        encoder_layer = nn.TransformerEncoderLayer(d_model=fusion_dim, nhead=nhead, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(fusion_dim, self.num_classes)

        self.to(device)

    def forward(self, z_cor,sentences_mask, binary_masks,mask_labels=None):
        B = len(sentences_mask)
        enc_mask = self.tokenizer(
            sentences_mask, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        text_outputs = self.text_encoder(**enc_mask)
        hidden_states = text_outputs.last_hidden_state  

        mask_token_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        mask_positions = (enc_mask["input_ids"] == mask_token_id)  

        binary_masks = binary_masks.unsqueeze(1)
        binary_masks = binary_masks*z_cor
        mask_feats_batch = self.mask_encoder(binary_masks.float().to(self.device))  

        logits_list = []  
        labels_list = []  
        total_logits = []
        total_labels = []

        for i in range(B):
            pos = mask_positions[i].nonzero(as_tuple=True)[0] 
            sentence_logits = []
            sentence_labels = []

            for j, p in enumerate(pos):
                left = max(0, p - 2)
                right = min(hidden_states.size(1), p + 3)  
                context_vec = hidden_states[i, left:right].mean(dim=0)  

                fused_input = torch.cat([context_vec, mask_feats_batch[i]], dim=-1) 
                fused_input = fused_input.unsqueeze(0).unsqueeze(0)
                transformed = self.transformer_encoder(fused_input)
                transformed = transformed.squeeze(0).squeeze(0)      

                logit = self.classifier(transformed) 
                sentence_logits.append(logit)
                total_logits.append(logit)

                if mask_labels is not None:
                    label = mask_labels[i][j]
                    sentence_labels.append(label)
                    total_labels.append(label)

            if len(sentence_logits) > 0:
                logits_list.append(torch.stack(sentence_logits))
            else:
                logits_list.append(torch.empty(0, self.num_classes, device=self.device))

            if mask_labels is not None:
                labels_list.append(torch.tensor(sentence_labels, dtype=torch.long, device=self.device))

        loss = None


        if mask_labels is not None and len(total_logits) > 0:
            total_logits_tensor = torch.stack(total_logits)
            total_labels_tensor = torch.tensor(total_labels, dtype=torch.long, device=self.device)
            loss = F.cross_entropy(total_logits_tensor, total_labels_tensor)

        else:
            loss = torch.tensor(0.0, device=self.device)

        return loss, logits_list, labels_list
    
def mask_objects_in_sentence(sentence, vocab):
    sentence_lower = sentence.lower()
    masked_sentence = sentence
    mask_labels = []

    matches = []
    for i, word in enumerate(vocab):
        word = word.lower().strip()
        if not word:
            continue

        pattern = r"\b" + re.escape(word) + r"\b"
        for m in re.finditer(pattern, sentence_lower):
            matches.append((m.start(), m.end(), i, word))

    matches = sorted(matches, key=lambda x: x[0], reverse=True)

    for start, end, idx, word in matches:
        masked_sentence = masked_sentence[:start] + "[MASK]" + masked_sentence[end:]
        mask_labels.append(idx)

    return masked_sentence, mask_labels

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = ["car", "tree", "stop sign", "dog", "cat", "bus", "truck", "person", "bike", "van"]
    word2id = {w: i for i, w in enumerate(vocab)}

    model = MaskedObjectPredictor(vocab=vocab, device=device)

    sentences = ["A [MASK] is parked near a [MASK]"]
    binary_masks = torch.randint(0, 2, (1, 16,32)) 

    mask_labels = torch.tensor([word2id["car"]])

    loss, logits = model(sentences, binary_masks, mask_labels)
    print("loss:", loss.item())
    print("logits shape:", logits.shape) 
