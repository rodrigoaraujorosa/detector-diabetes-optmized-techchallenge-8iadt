# ⚙️ Como Executar o Projeto Localmente

## Pré-requisitos

- Python **3.10** ou superior
- **pip** (gerenciador de pacotes Python)
- Chave de API da **OpenAI** (opcional — necessária apenas para a análise por IA Generativa)

---

## Passo a Passo

### 1. Clone o repositório e acesse a pasta do projeto

```bash
git clone <url-do-repositorio>
cd <nome-da-pasta>
```

### 2. Crie e ative um ambiente virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Atualize o pip (opcional)

```bash
python -m pip install --upgrade pip
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

### 5. Configure a chave da OpenAI

Copie o arquivo de exemplo e adicione sua chave:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Edite o arquivo `.env` e substitua o placeholder pela sua chave real:

```
OPENAI_API_KEY=sk-SuaChaveAqui
```

> **Obs.:** Este passo é opcional. Sem a chave configurada, a aplicação funciona normalmente — apenas a análise por IA Generativa ficará desabilitada.

### 6. Selecione o interpretador Python no VSCode (opcional)

1. Pressione `Ctrl+Shift+P`
2. Digite **"Python: Select Interpreter"**
3. Selecione o `.venv` do projeto (ex: `.venv\Scripts\python.exe`)

### 7. Execute o Gradio

```bash
python app.py
```

A interface será aberta automaticamente no navegador em `http://localhost:7860`.
