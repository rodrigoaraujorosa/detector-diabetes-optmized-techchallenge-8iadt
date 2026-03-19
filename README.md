---
title: Diagnóstico de Diabetes - Random Forest - Otimizado
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: true
license: mit
short_description: Suporte ao diagnóstico de diabetes usando Machine Learning
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/69239245f2835e7820b31ade/eBdPnxehVjii957k4G5zA.png
---

# 🏥 Diagnóstico de Diabetes - Random Forest - Otimizado

Sistema de suporte ao diagnóstico de diabetes utilizando **Machine Learning (Random Forest)** baseado no dataset Pima Indians Diabetes Database.

> **⚠️ IMPORTANTE**: Esta é uma aplicação educacional desenvolvida como projeto acadêmico da FIAP. 
> **NÃO substitui avaliação médica profissional**. Sempre consulte um médico para diagnóstico e tratamento.

## 🎯 Funcionalidades

- ✅ Predição de diabetes baseada em parâmetros clínicos
- 📊 Cálculo de probabilidades (Diabético/Não Diabético)
- 📋 Interpretação automática dos parâmetros inseridos
- 🤖 **Análise por IA Generativa (OpenAI GPT)** — explicações em linguagem natural e insights acionáveis para médicos
- 🎨 Interface intuitiva com controles deslizantes
- 💡 Exemplos pré-configurados para teste

## 🧠 Modelo (Otimizado via Algoritmo Genético)

- **Algoritmo**: Random Forest Classifier
- **Hiperparâmetros Otimizados**:
  - `n_estimators`: 43
  - `max_depth`: 19
  - `min_samples_leaf`: 1
  - `min_samples_split`: 5
  - `max_features`: 'log2'
- **Acurácia**: ~79%
  
## 📊 Dataset

**Pima Indians Diabetes Database** (National Institute of Diabetes and Digestive and Kidney Diseases - NIH)

**Características:**
- **População**: Mulheres com ascendência indígena Pima
- **Idade mínima**: 21 anos
- **Total de registros**: 768 pacientes
- **Features**: 8 parâmetros clínicos

### Parâmetros de Entrada:

1. **Pregnancies** (Gestações): Número de vezes que a paciente engravidou
2. **Glucose** (Glicose): Concentração de glicose plasmática (mg/dL) após 2 horas em teste OGTT
   - < 140: Normal
   - 140-199: Pré-diabetes
   - ≥ 200: Diabetes
3. **BloodPressure** (Pressão Arterial): Pressão arterial diastólica (mm Hg)
4. **SkinThickness** (Espessura da Pele): Espessura da dobra cutânea do tríceps (mm)
5. **Insulin** (Insulina): Nível de insulina sérica (mu U/ml)
6. **BMI** (IMC): Índice de Massa Corporal (peso kg / altura m²)
7. **DiabetesPedigreeFunction**: Função de pedigree de diabetes (histórico familiar)
8. **Age** (Idade): Idade em anos

## 🚀 Como Usar

1. **Ajuste os parâmetros** usando os controles deslizantes:
   - Gestações, Glicose, Pressão Arterial, etc.
2. **Clique em "Submit"** ou use um dos exemplos pré-configurados
3. **Visualize o resultado**:
   - Predição (Diabético/Não Diabético)
   - Probabilidades calculadas
   - Interpretação dos parâmetros
   - **Análise clínica gerada por IA** (requer `OPENAI_API_KEY`)

## 📝 Exemplos Inclusos

A aplicação inclui 5 casos de teste:
- ✅ Baixo risco de diabetes
- ⚠️ Risco moderado
- 🔴 Alto risco de diabetes
- 👶 Paciente jovem
- 📊 Diversos perfis clínicos

## 🛠️ Tecnologias

- **Python** 3.10+
- **Gradio** - Interface web interativa
- **Scikit-learn** - Modelo Random Forest
- **OpenAI GPT-4o-mini** - Explicações em linguagem natural e insights clínicos
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Pickle** - Serialização do modelo

## 🔑 Configuração da API OpenAI (Hugging Face Space)

Para habilitar a análise por IA Generativa:

1. Acesse as **Settings** do seu Space no Hugging Face
2. Vá em **Variables and secrets**
3. Adicione um novo secret com o nome: `OPENAI_API_KEY`
4. Cole sua chave de API da OpenAI como valor

> Sem a chave configurada, as demais funcionalidades (predição e interpretação dos parâmetros) continuam funcionando normalmente.

## 📚 Projeto Acadêmico

**Instituição**: FIAP  
**Curso**: IA para DEVs - 8IADT  
**Fase**: 2  
**Projeto**: Tech Challenge - Otimização de Hiperparametros via Algoritmo Genético  
**Grupo**: 61

### 👥 Integrantes:
- Rodrigo de Araújo Rosa
- Elias Maximiano da Silva
- Fábia Gomes de Jesus
- Danilo Pereira

### 🎯 Objetivo do Projeto:
Construir uma solução inicial com foco em IA para processamento de exames médicos, 
aplicando fundamentos de Machine Learning para criar um sistema de suporte ao diagnóstico e otimizar o modelo com base em algoritmos genéticos, bem como integrar com uma LLM para Interpretação de Resultados.

## 📖 Referências

- [NIH - Pima Indians Diabetes Database](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/)
- [Glucose Tolerance Test - NIH](https://www.ncbi.nlm.nih.gov/books/NBK532915/)
- [Diabetes and Hypertension - ADA](https://diabetesjournals.org/care/article/40/9/1273/36772/)

## 📄 Licença

MIT License - Uso livre para fins acadêmicos e educacionais.

## 📧 Contato

Para dúvidas ou feedback sobre o projeto:
- **Email**: rodrigoaraujo.r@gmail.com
- **LinkedIn**: [Rodrigo de Araujo Rosa](https://www.linkedin.com/in/rodrigoaraujorosa/)

---

**Desenvolvido para aprendizado em IA e Machine Learning**