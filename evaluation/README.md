# Avaliacao do RAG

Esta pasta guarda o gabarito usado para avaliar o chatbot. O arquivo `gabarito_avaliacao.json` nao alimenta o CBR e nao e consultado pelo chatbot durante uma conversa normal.

As perguntas foram organizadas para cobrir os principais documentos locais:

- PPC e estrutura curricular;
- TCC, anexos e formularios;
- estagio;
- atividades complementares;
- regulamento geral da graduacao;
- calendario universitario;
- apresentacao e orientacoes de template.

Campos do gabarito:

- `question`: pergunta enviada ao chatbot;
- `ground_truth`: resposta de referencia resumida;
- `expected_sources`: documentos que deveriam aparecer entre as fontes recuperadas.

Para executar somente a recuperacao e salvar respostas/contextos:

```bash
python3 scripts/evaluate_ragas.py --skip-ragas
```

Para lotes maiores, use pausa entre perguntas e continue mesmo quando a API retornar rate limit:

```bash
python3 scripts/evaluate_ragas.py --skip-ragas --delay-seconds 5 --continue-on-error
```

Para executar com metricas Ragas:

```bash
pip install -r requirements-eval.txt
python3 scripts/evaluate_ragas.py
```
