# CustomTextGenerator

## Introdução

`CustomTextGenerator` é uma classe Python para treinamento e geração de texto usando o modelo GPT-2 da Hugging Face. Ela permite que você treine um modelo GPT-2 em um conjunto de dados personalizado e gere texto com base em entradas fornecidas. O treinamento é feito com o `datasets` e o `transformers` da Hugging Face, e o modelo pode ser ajustado para gerar texto com diferentes comprimentos e estilos.

## Requisitos

- Python 3.6 ou superior
- `pandas`
- `torch`
- `transformers`
- `sklearn`
- `datasets`

## Instalação

Você pode instalar as dependências necessárias usando pip. Execute o seguinte comando para instalar as bibliotecas:

```bash
pip install pandas torch transformers scikit-learn datasets
```

## Uso

1. **Preparação dos Dados**

   Certifique-se de ter um DataFrame do Pandas com uma coluna de texto que deseja usar para treinar o modelo. A coluna deve conter textos que você deseja gerar ou adaptar com o modelo GPT-2.

2. **Criação e Treinamento do Modelo**

   Crie uma instância da classe `CustomTextGenerator` e forneça o DataFrame e o nome da coluna de texto. Em seguida, chame o método `train()` para treinar o modelo.

3. **Geração de Texto**

   Após o treinamento, você pode gerar texto com base em uma entrada fornecida usando o método `generate_text()`.

### Exemplo de Código

Aqui está um exemplo completo de como usar a classe `CustomTextGenerator`:

```python
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import Dataset

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
    
    def prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        # Prepare os dados para treinamento
        def tokenize_function(examples):
            return self.tokenizer(
                examples[self.text_column], 
                padding=False,         # Não use padding
                truncation=True,       # Truncar se exceder o comprimento máximo
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

# Exemplo de uso
if __name__ == "__main__":
    # Supondo que você tenha um DataFrame df com a coluna 'translation_text'
    df = pd.DataFrame({
        'translation_text': [
            'A tecnologia está evoluindo rapidamente.',
            'Os avanços em IA estão mudando o mundo.',
            'O futuro é incerto, mas cheio de possibilidades.',
            'Data science é uma área em crescimento.',
            'Aprender novas habilidades é sempre benéfico.'
        ]
    })
    
    # Crie uma instância da classe CustomTextGenerator
    text_gen = CustomTextGenerator(dataframe=df, text_column='translation_text', model_name='gpt2', max_length=100)
    
    # Treine o modelo
    text_gen.train()
    
    # Gere texto a partir de uma entrada
    generated_text = text_gen.generate_text("A tecnologia está evoluindo")
    print(generated_text)
```

### Descrição dos Métodos

- **`__init__(self, dataframe, text_column, model_name, max_length, train_size)`**: Inicializa a classe com o DataFrame, coluna de texto, nome do modelo, comprimento máximo para geração de texto e tamanho de treinamento.

- **`_add_padding_token(self)`**: Adiciona um token de padding se necessário. O GPT-2 não requer padding para geração, mas o método garante que o token de padding esteja disponível para outras operações.

- **`prepare_dataset(self, df)`**: Prepara os dados para treinamento, incluindo tokenização e truncamento.

- **`train(self)`**: Treina o modelo usando o método `Trainer` da Hugging Face.

- **`generate_text(self, input_text, max_length)`**: Gera texto com base na entrada fornecida.

## Contribuição

Contribuições são bem-vindas! Se você encontrar algum problema ou tiver sugestões de melhorias, por favor, abra um issue ou um pull request.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
