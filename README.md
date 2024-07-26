# Tutorial: Gerador de Texto Personalizado com GPT-2

## Introdução

Este tutorial explica como usar a classe `CustomTextGenerator` para treinar um modelo de linguagem GPT-2 com um conjunto de dados personalizado e gerar texto com ele. A classe é baseada na biblioteca `transformers` da Hugging Face e usa PyTorch para treinamento e geração de texto.

## Pré-requisitos

Antes de começar, certifique-se de que você tem as seguintes bibliotecas instaladas:

- `pandas`
- `torch`
- `transformers`
- `sklearn`
- `datasets`

Você pode instalar essas bibliotecas usando o seguinte comando:

```bash
pip install pandas torch transformers scikit-learn datasets
```

## Código

### Importando Bibliotecas

Comece importando as bibliotecas necessárias:

```python
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
```

### Definindo a Classe `CustomTextGenerator`

Aqui está o código da classe `CustomTextGenerator`:

```python
class CustomTextGenerator:
    def __init__(self, dataframe: pd.DataFrame, text_column: str, model_name='gpt2', max_length=50, train_size=0.8):
        self.dataframe = dataframe
        self.text_column = text_column
        self.model_name = model_name
        self.max_length = max_length
        self.train_size = train_size
        
        # Inicialize o tokenizer e o modelo
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Adiciona um token de padding, se necessário
        self._add_padding_token()
        
        # Divida o DataFrame em conjuntos de treinamento e teste
        self.train_df, self.val_df = train_test_split(self.dataframe, train_size=self.train_size, shuffle=True)
        
        # Prepare os dados para treinamento
        self.train_dataset = self.prepare_dataset(self.train_df)
        self.val_dataset = self.prepare_dataset(self.val_df)
        
        # Configurações de treinamento
        self.training_args = TrainingArguments(
            output_dir='./results',          # Diretório para salvar os resultados
            num_train_epochs=1,              # Número de épocas
            per_device_train_batch_size=4,   # Tamanho do lote de treinamento
            per_device_eval_batch_size=4,    # Tamanho do lote de avaliação
            warmup_steps=500,                # Número de passos para o aquecimento
            weight_decay=0.01,               # Decaimento do peso
            logging_dir='./logs',            # Diretório para salvar logs
            logging_steps=10,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer
        )
    
    def _add_padding_token(self):
        # Adiciona um token de padding se não existir
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        # Atualize o modelo com o novo token de padding
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        # Prepare os dados para treinamento
        def tokenize_function(examples):
            return self.tokenizer(
                examples[self.text_column], 
                padding='max_length',         # Use padding para garantir comprimento fixo
                truncation=True,              # Truncar se exceder o comprimento máximo
                max_length=self.max_length
            )
        
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(tokenize_function, batched=True)
        return dataset
    
    def train(self):
        # Treine o modelo
        self.trainer.train()
    
    def generate_text(self, input_text: str, max_length: int = None) -> str:
        if max_length is None:
            max_length = self.max_length
        
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.7,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Exemplo de Uso

Para usar a classe `CustomTextGenerator`, siga estes passos:

1. **Prepare o DataFrame**: Crie um DataFrame com a coluna de texto que deseja usar para treinar o modelo.

    ```python
    df = pd.DataFrame({
        'translation_text': [
            'A tecnologia está evoluindo rapidamente.',
            'Os avanços em IA estão mudando o mundo.',
            'O futuro é incerto, mas cheio de possibilidades.',
            'Data science é uma área em crescimento.',
            'Aprender novas habilidades é sempre benéfico.'
        ]
    })
    ```

2. **Crie uma Instância da Classe**: Inicialize a classe `CustomTextGenerator` com o DataFrame e outras configurações.

    ```python
    text_gen = CustomTextGenerator(dataframe=df, text_column='translation_text', model_name='gpt2', max_length=100)
    ```

3. **Treine o Modelo**: Execute o treinamento do modelo.

    ```python
    text_gen.train()
    ```

4. **Gere Texto**: Gere texto a partir de uma entrada.

    ```python
    generated_text = text_gen.generate_text("A tecnologia está evoluindo")
    print(generated_text)
    ```

### Conclusão

Este tutorial fornece uma visão geral de como usar a classe `CustomTextGenerator` para treinar um modelo GPT-2 com um conjunto de dados personalizado e gerar texto. Siga as etapas acima para configurar e executar seu gerador de texto.

Se você encontrar algum problema ou tiver dúvidas, não hesite em buscar ajuda na documentação da [Hugging Face](https://huggingface.co/docs/transformers) ou na comunidade de suporte.
