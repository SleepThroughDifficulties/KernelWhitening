from transformers import AutoConfig, AutoTokenizer

config = AutoConfig.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

s1 = 'i love dad'
s2 = 'you love me'

encoding = tokenizer.encode_plus(s1, s2, max_length=768, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
token_type_ids = encoding['token_type_ids']
print(1)
twosentence = False
if token_type_ids:
    p = token_type_ids.index(1)

    twosentence = True
    input_ids1 = input_ids[0:p]
    attention_mask1 = attention_mask[0:p]
    token_type_ids1 = token_type_ids[0:p]

    input_ids2 = input_ids[0:1] + input_ids[p:]
    attention_mask2 = attention_mask[0:1] + attention_mask[p:]
    token_type_ids2 = [1] + token_type_ids[p:]
print(1)