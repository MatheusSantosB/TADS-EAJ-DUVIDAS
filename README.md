# Chatbot TADS

Projeto inicial de chatbot academico para responder perguntas sobre o curso de TADS usando documentos locais.

Tecnologias usadas:

- Python
- LangChain
- LangGraph
- RAG
- Groq como LLM
- Chroma como banco vetorial local
- CBR simples com `cases/cases.json`
- Execucao no terminal

## Estrutura

```text
tads_chatbot/
  main.py
  .env.example
  requirements.txt
  README.md
  data/
  cases/
    cases.json
  src/
    __init__.py
    config.py
    ingestion.py
    retriever.py
    llm.py
    prompts.py
    cbr.py
    graph.py
    utils.py
```

## 1. Criar ambiente virtual

No terminal, entre na pasta do projeto:

```bash
cd /mnt/c/TADS-EAJ-DUVIDAS/tads_chatbot
```

Crie e ative o ambiente virtual. No WSL, para evitar lentidao em pastas do Windows, uma boa opcao e manter o ambiente virtual na home do Linux:

```bash
python3 -m venv /home/matheus/.venvs/tads_chatbot
source /home/matheus/.venvs/tads_chatbot/bin/activate
```

No Windows PowerShell, a ativacao seria:

```powershell
cd "C:\TADS-EAJ-DUVIDAS\tads_chatbot"
py -m venv .venv
.venv\Scripts\Activate.ps1
```

## 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Observacao: na primeira execucao, o modelo de embeddings pode ser baixado automaticamente pelo `fastembed`.

## 3. Configurar arquivo .env

Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

Edite o `.env` e coloque sua chave da Groq:

```env
GROQ_API_KEY=sua_chave_da_groq_aqui
```

Voce pode trocar o modelo no mesmo arquivo:

```env
GROQ_MODEL=llama-3.1-8b-instant
```

## 4. Colocar documentos na pasta data

Coloque arquivos `.pdf` ou `.txt` dentro da pasta:

```text
tads_chatbot/data/
```

Exemplo, se quiser usar os documentos ja existentes do curso:

```bash
cp "../Documentos do curso"/*.pdf data/
```

O projeto inicial suporta PDF e TXT. Arquivos `.doc` e `.docx` nao sao indexados nesta versao.

## 5. Rodar a ingestao

Execute:

```bash
python3 main.py
```

Escolha:

```text
1 - Rodar ingestao dos documentos
```

Isso cria o banco vetorial local em:

```text
chroma_db/
```

Durante a ingestao, o programa mostra o arquivo atual, quantos chunks foram
gerados e quantos chunks ja foram inseridos no Chroma. Isso ajuda principalmente
com PDFs maiores, como o PPC do curso. Paginas sem texto extraivel sao
ignoradas e, se algum arquivo falhar, o nome do arquivo aparece no resumo final.

Se precisar reduzir ou aumentar o tamanho do lote de insercao, ajuste no `.env`:

```env
INGESTION_BATCH_SIZE=100
```

## 6. Executar o chatbot no terminal

Depois da ingestao, rode novamente:

```bash
python3 main.py
```

Escolha:

```text
2 - Iniciar chatbot
```

Digite perguntas como:

```text
Quais sao as regras para o TCC?
Como funciona o estagio?
Quais documentos falam sobre atividades complementares?
```

Para sair:

```text
sair
```

## Como funciona

O fluxo do LangGraph tem cinco nos simples:

1. `classify_question`: estima a categoria da pergunta.
2. `retrieve_docs`: busca trechos relevantes no Chroma.
3. `retrieve_cases`: procura um caso parecido em `cases/cases.json`.
4. `generate_answer`: usa Groq para responder com base nos documentos.
5. `final_response`: salva a nova pergunta como caso, quando houver fontes.

O CBR nao substitui os documentos. Ele apenas ajuda a resposta quando encontra uma pergunta anterior parecida.

## Comportamento esperado

O chatbot:

- responde em portugues do Brasil;
- usa os documentos locais como base;
- mostra as fontes usadas;
- evita inventar informacoes;
- informa quando nao encontra base suficiente;
- roda somente no terminal.

## Observacoes

Se aparecer erro dizendo que `GROQ_API_KEY` nao foi encontrada, confira se o arquivo `.env` existe na raiz do projeto e se a chave foi preenchida.

Se o chatbot nao encontrar documentos, rode a ingestao novamente depois de colocar PDFs ou TXTs na pasta `data/`.
