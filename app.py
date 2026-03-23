"""
Aplicação Gradio para Diagnóstico de Diabetes
Autor: Grupo 61 - FIAP 8IADT
Baseado no dataset Pima Indians Diabetes Database
"""

import gradio as gr
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Carrega variáveis do arquivo .env para uso local (no Hugging Face usa Secrets)
load_dotenv()

# Carregar modelo
print("📦 Carregando modelo de diabetes...")
try:
    with open('models/diabetes_random_forest_model_optimized.pkl', 'rb') as f:
        modelo = pickle.load(f)
    print("✅ Modelo otimizado por algoritmo genético carregado com sucesso!")
except FileNotFoundError:
    print("❌ Erro: Arquivo 'models/diabetes_random_forest_model_optimized.pkl' não encontrado!")
    raise

def gerar_explicacao_llm(dados_paciente: dict, predicao: int, prob_diabetico: float) -> str:
    """
    Gera explicação em linguagem natural do diagnóstico usando OpenAI GPT.
    Transforma dados numéricos em insights acionáveis para médicos.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return (
            "⚠️ **Chave da API OpenAI não configurada.**\n\n"
            "Para habilitar explicações por IA, configure a variável de ambiente "
            "`OPENAI_API_KEY` nas **Settings > Secrets** do seu Hugging Face Space "
            "ou no ambiente local."
        )

    status = "DIABÉTICO" if predicao == 1 else "NÃO DIABÉTICO"
    confianca = prob_diabetico * 100 if predicao == 1 else (1 - prob_diabetico) * 100

    # Estrutura do prompt preparada para receber dados textuais no Módulo 3
    prompt_sistema = (
        "Você é um assistente médico especializado em diabetes, com expertise em "
        "interpretar dados clínicos e fornecer insights acionáveis para profissionais "
        "de saúde. Suas respostas devem ser claras, precisas e clinicamente relevantes. "
        "Responda sempre em português brasileiro."
    )

    prompt_usuario = f"""Analise os dados clínicos abaixo e o resultado de um modelo de Machine Learning (Random Forest) treinado no Pima Indians Diabetes Database.

**Dados Clínicos do Paciente:**
- Número de gestações: {dados_paciente['pregnancies']}
- Glicose plasmática: {dados_paciente['glucose']} mg/dL
- Pressão arterial diastólica: {dados_paciente['blood_pressure']} mm Hg
- Espessura da dobra cutânea (tríceps): {dados_paciente['skin_thickness']} mm
- Insulina sérica: {dados_paciente['insulin']} mu U/ml
- IMC: {dados_paciente['bmi']:.1f}
- Função de pedigree de diabetes: {dados_paciente['diabetes_pedigree']:.3f}
- Idade: {dados_paciente['age']} anos

**Resultado do Modelo:**
- Diagnóstico: **{status}**
- Probabilidade de diabetes: {prob_diabetico * 100:.1f}%
- Confiança do diagnóstico: {confianca:.1f}%

Forneça uma análise estruturada com os seguintes tópicos:

1. **Explicação do Diagnóstico**: Explique em linguagem clara o resultado, destacando quais parâmetros mais influenciaram a predição.

2. **Fatores de Risco Identificados**: Liste os principais fatores de risco presentes nos dados do paciente.

3. **Insights Acionáveis para o Médico**: De 3 a 5 recomendações práticas e específicas (exames adicionais, monitoramento, ajustes de conduta, etc.).

4. **Próximos Passos Sugeridos**: Indique as próximas etapas recomendadas para acompanhamento do paciente.

> *Lembrete: este é um modelo preditivo e não substitui avaliação médica completa.*"""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        return response.choices[0].message.content # type: ignore
    except Exception as e:
        return f"⚠️ **Erro ao gerar explicação por IA:** {e}"


ARQUIVO_CONSULTAS = "consultas.txt"

def salvar_consulta(dados_paciente: dict, predicao: int, prob_diabetico: float, explicacao_llm: str) -> None:
    """
    Salva os dados da consulta e a explicação da LLM em arquivo txt,
    incrementando o arquivo a cada nova consulta.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "DIABÉTICO" if predicao == 1 else "NÃO DIABÉTICO"
    confianca = prob_diabetico * 100 if predicao == 1 else (1 - prob_diabetico) * 100

    separador = "=" * 80
    entrada = (
        f"\n{separador}\n"
        f"CONSULTA REALIZADA EM: {timestamp}\n"
        f"{separador}\n"
        f"DADOS DO PACIENTE:\n"
        f"  Gestações              : {dados_paciente['pregnancies']}\n"
        f"  Glicose                : {dados_paciente['glucose']} mg/dL\n"
        f"  Pressão Arterial       : {dados_paciente['blood_pressure']} mm Hg\n"
        f"  Espessura da Pele      : {dados_paciente['skin_thickness']} mm\n"
        f"  Insulina               : {dados_paciente['insulin']} mu U/ml\n"
        f"  IMC                    : {dados_paciente['bmi']:.1f}\n"
        f"  Função Pedigree        : {dados_paciente['diabetes_pedigree']:.3f}\n"
        f"  Idade                  : {dados_paciente['age']} anos\n"
        f"\nRESULTADO DO MODELO:\n"
        f"  Diagnóstico            : {status}\n"
        f"  Probabilidade diabetes : {prob_diabetico * 100:.1f}%\n"
        f"  Confiança              : {confianca:.1f}%\n"
        f"\nANÁLISE GERADA PELA IA:\n"
        f"{explicacao_llm}\n"
    )

    with open(ARQUIVO_CONSULTAS, "a", encoding="utf-8") as f:
        f.write(entrada)


def prever_diabetes(pregnancies, glucose, blood_pressure, skin_thickness,
                   insulin, bmi, diabetes_pedigree, age):
    """
    Prediz se um paciente tem diabetes com base em parâmetros clínicos.

    Parâmetros:
    -----------
    pregnancies : int - Número de gestações
    glucose : float - Concentração de glicose plasmática (mg/dL)
    blood_pressure : float - Pressão arterial diastólica (mm Hg)
    skin_thickness : float - Espessura da dobra cutânea do tríceps (mm)
    insulin : float - Nível de insulina sérica (mu U/ml)
    bmi : float - Índice de Massa Corporal (peso em kg/(altura em m)^2)
    diabetes_pedigree : float - Função de pedigree de diabetes
    age : int - Idade em anos

    Retorna:
    --------
    resultado, probabilidades, interpretação, explicacao_llm
    """
    
    # Validações básicas
    if glucose < 0 or glucose > 300:
        return "⚠️ **Erro**: Glicose deve estar entre 0 e 300 mg/dL", "", "", ""
    
    if bmi < 0 or bmi > 70:
        return "⚠️ **Erro**: IMC deve estar entre 0 e 70", "", "", ""
    
    if age < 21:
        return "⚠️ **Nota**: Este modelo foi treinado com pacientes com 21 anos ou mais", "", "", ""
    
    # Criar dataframe com os dados
    dados = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    
    # Fazer previsão
    predicao = modelo.predict(dados)[0]
    probabilidades = modelo.predict_proba(dados)[0]
    
    prob_nao_diabetico = float(probabilidades[0])
    prob_diabetico = float(probabilidades[1])
    
    # Resultado principal
    if predicao == 1:
        resultado = f"⚠️ **DIABÉTICO**\n\n**Confiança:** {prob_diabetico*100:.1f}%"
        emoji = "🔴"
    else:
        resultado = f"✅ **NÃO DIABÉTICO**\n\n**Confiança:** {prob_nao_diabetico*100:.1f}%"
        emoji = "🟢"
    
    # Probabilidades
    prob_texto = f"""
## 📊 Probabilidades:
- **Não Diabético:** {prob_nao_diabetico*100:.1f}%
- **Diabético:** {prob_diabetico*100:.1f}%
"""
    
    # Interpretação dos dados
    interpretacao = f"""
## 📋 Análise dos Parâmetros:

### Glicose: {glucose} mg/dL
"""
    
    if glucose < 140:
        interpretacao += "- ✅ **Normal** (< 140 mg/dL)\n"
    elif glucose < 200:
        interpretacao += "- ⚠️ **Pré-diabetes** (140-199 mg/dL)\n"
    else:
        interpretacao += "- 🔴 **Diabético** (≥ 200 mg/dL)\n"
    
    interpretacao += f"\n### IMC: {bmi:.1f}\n"
    if bmi < 18.5:
        interpretacao += "- Abaixo do peso\n"
    elif bmi < 25:
        interpretacao += "- ✅ Peso normal\n"
    elif bmi < 30:
        interpretacao += "- ⚠️ Sobrepeso\n"
    else:
        interpretacao += "- 🔴 Obesidade\n"
    
    interpretacao += f"\n### Idade: {age} anos\n"
    interpretacao += f"### Gestações: {pregnancies}\n"
    interpretacao += f"### Pressão Arterial: {blood_pressure} mm Hg\n"
    
    interpretacao += """
---
**⚕️ Nota Importante:** Este é um modelo preditivo baseado em Machine Learning
e **NÃO substitui** uma avaliação médica profissional. Consulte sempre um médico
para diagnóstico e tratamento adequados.
"""

    # Geração da explicação via LLM
    dados_paciente = {
        "pregnancies": pregnancies,
        "glucose": glucose,
        "blood_pressure": blood_pressure,
        "skin_thickness": skin_thickness,
        "insulin": insulin,
        "bmi": bmi,
        "diabetes_pedigree": diabetes_pedigree,
        "age": age,
    }
    explicacao_llm = gerar_explicacao_llm(dados_paciente, predicao, prob_diabetico)
    salvar_consulta(dados_paciente, predicao, prob_diabetico, explicacao_llm)

    return resultado, prob_texto, interpretacao, explicacao_llm

# Exemplos de casos para teste
exemplos = [
    [6, 155, 80, 35, 85, 33.6, 0.627, 50],  # Alto risco
    [1, 85, 49, 33, 134, 24, 0.351, 31],   # Baixo risco
    [8, 188, 76, 25, 109, 23.3, 0.672, 32],   # Risco moderado
    [1, 89, 66, 23, 94, 27, 0.167, 21],  # Jovem baixo risco
    [0, 137, 40, 35, 168, 43.1, 2.288, 33] # Risco elevado
]

# Interface Gradio
interface = gr.Interface(
    fn=prever_diabetes,
    inputs=[
        gr.Slider(minimum=0, maximum=17, step=1, value=1, label="🤰 Gestações"),
        gr.Slider(minimum=0, maximum=200, step=1, value=141, label="🩸 Glicose (mg/dL)"),
        gr.Slider(minimum=0, maximum=122, step=1, value=43, label="💓 Pressão Arterial Diastólica (mm Hg)"),
        gr.Slider(minimum=0, maximum=99, step=1, value=25, label="📏 Espessura da Pele (mm)"),
        gr.Slider(minimum=0, maximum=846, step=1, value=90, label="💉 Insulina (mu U/ml)"),
        gr.Slider(minimum=0, maximum=67, step=0.1, value=23.3, label="⚖️ IMC"),
        gr.Slider(minimum=0.0, maximum=2.5, step=0.001, value=0.672, label="🧬 Função Pedigree de Diabetes"),
        gr.Slider(minimum=21, maximum=81, step=1, value=32, label="👤 Idade (anos)")
    ],
    outputs=[
        gr.Markdown(label="🎯 Resultado da Predição"),
        gr.Markdown(label="📊 Probabilidades"),
        gr.Markdown(label="📋 Interpretação dos Parâmetros"),
        gr.Markdown(label="🤖 Análise por IA (OpenAI GPT)"),
    ],
    title="🏥 Diagnóstico de Diabetes - Random Forest + IA Generativa",
    description="""
    ### Sistema de Suporte ao Diagnóstico de Diabetes com IA Generativa

    Esta aplicação combina **Machine Learning (Random Forest)** com **IA Generativa (OpenAI GPT)**
    para predizer diabetes e gerar explicações clínicas em linguagem natural.
    **Apenas para uso acadêmico e informativo.**

    **📚 Baseado no dataset:** Pima Indians Diabetes Database (NIH)

    **🎯 Como funciona:**
    1. Ajuste os parâmetros clínicos do paciente usando os controles deslizantes
    **💡 Dica:** Experimente os exemplos abaixo para ver diferentes cenários!
    2. O modelo Random Forest calcula a probabilidade de diabetes
    3. O GPT (OpenAI) interpreta o resultado e gera insights acionáveis para o médico

    ---
    **👥 Desenvolvido por:** Grupo 61
    **🎓 Projeto:** IA para DEVs - Tech Challenge Fase 2 - Diagnóstico de Doenças com otimização do modelo via algoritmo genético
    """,
    examples=exemplos,
    examples_per_page=5,
    flagging_mode="never"
)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 GRADIO - DIAGNÓSTICO DE DIABETES COM MODELO OTIMIZADO POR ALGORITMO GENÉTICO")
    print("="*80)
    print("\n💡 Interface será aberta no navegador")
    print("🏥 Grupo 61 - FIAP 8IADT Tech Challenge - Fase 2")
    print("="*80 + "\n")
    
    interface.launch(theme=gr.themes.Soft()) # type: ignore