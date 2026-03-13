"""
Aplicação Gradio para Detecção de Diabetes
Autor: Grupo 40 - FIAP 8IADT
Baseado no dataset Pima Indians Diabetes Database
"""

import gradio as gr
import pickle
import pandas as pd
import numpy as np

# Carregar modelo
print("📦 Carregando modelo de diabetes...")
try:
    with open('models/diabetes_random_forest_model_optimized.pkl', 'rb') as f:
        modelo = pickle.load(f)
    print("✅ Modelo carregado com sucesso!")
except FileNotFoundError:
    print("❌ Erro: Arquivo 'models/diabetes_random_forest_model_optimized.pkl' não encontrado!")
    raise

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
    resultado, probabilidades, interpretação
    """
    
    # Validações básicas
    if glucose < 0 or glucose > 300:
        return "⚠️ **Erro**: Glicose deve estar entre 0 e 300 mg/dL", "", ""
    
    if bmi < 0 or bmi > 70:
        return "⚠️ **Erro**: IMC deve estar entre 0 e 70", "", ""
    
    if age < 21:
        return "⚠️ **Nota**: Este modelo foi treinado com pacientes com 21 anos ou mais", "", ""
    
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
    
    interpretacao += f"""

    TODO: Alimentar uma LLM para gerar uma interpretação mais detalhada com base nos parâmetros clínicos do paciente.

---
**⚕️ Nota Importante:** Este é um modelo preditivo baseado em Machine Learning 
e **NÃO substitui** uma avaliação médica profissional. Consulte sempre um médico 
para diagnóstico e tratamento adequados.
"""
    return resultado, prob_texto, interpretacao

# Exemplos de casos para teste
exemplos = [
    [6, 148, 72, 35, 85, 33.6, 0.627, 50],  # Alto risco
    [1, 85, 66, 29, 120, 26.6, 0.351, 31],   # Baixo risco
    [8, 183, 64, 25, 90, 23.3, 0.672, 32],   # Risco moderado
    [1, 89, 66, 23, 94, 28.1, 0.167, 21],  # Jovem baixo risco
    [0, 137, 40, 35, 168, 43.1, 2.288, 33] # Risco elevado
]

# Interface Gradio
interface = gr.Interface(
    fn=prever_diabetes,
    inputs=[
        gr.Slider(minimum=0, maximum=17, step=1, value=1, label="🤰 Gestações"),
        gr.Slider(minimum=0, maximum=200, step=1, value=120, label="🩸 Glicose (mg/dL)"),
        gr.Slider(minimum=0, maximum=122, step=1, value=70, label="💓 Pressão Arterial Diastólica (mm Hg)"),
        gr.Slider(minimum=0, maximum=99, step=1, value=20, label="📏 Espessura da Pele (mm)"),
        gr.Slider(minimum=0, maximum=846, step=1, value=79, label="💉 Insulina (mu U/ml)"),
        gr.Slider(minimum=0, maximum=67, step=0.1, value=32.0, label="⚖️ IMC"),
        gr.Slider(minimum=0.0, maximum=2.5, step=0.001, value=0.5, label="🧬 Função Pedigree de Diabetes"),
        gr.Slider(minimum=21, maximum=81, step=1, value=33, label="👤 Idade (anos)")
    ],
    outputs=[
        gr.Markdown(label="🎯 Resultado da Predição"),
        gr.Markdown(label="📊 Probabilidades"),
        gr.Markdown(label="📋 Interpretação")
    ],
    title="🏥 Detector de Diabetes - Random Forest",
    description="""
    ### Sistema de Suporte ao Diagnóstico de Diabetes
    
    Esta aplicação utiliza **Machine Learning (Random Forest)** para predizer a probabilidade 
    de diabetes com base em parâmetros clínicos de pacientes. **Apenas para uso acadêmico e informativo.**
    
    **📚 Baseado no dataset:** Pima Indians Diabetes Database (NIH)
    
    **🎯 Como funciona:**
    1. Ajuste os parâmetros clínicos do paciente usando os controles deslizantes
    2. O modelo Random Forest analisa os dados e calcula a probabilidade
    3. Visualize o resultado, probabilidades e interpretação dos parâmetros
    
    **💡 Dica:** Experimente os exemplos abaixo para ver diferentes cenários!
    
    ---
    **👥 Desenvolvido por:** Rodrigo Rosa - Grupo 40 - FIAP 8IADT Tech Challenge  
    **🎓 Projeto:** IA para DEVs - Fase 1 - Diagnóstico de Doenças
    """,
    examples=exemplos,
    examples_per_page=5,
    flagging_mode="never"
)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 GRADIO - DETECTOR DE DIABETES")
    print("="*80)
    print("\n💡 Interface será aberta no navegador")
    print("🏥 Grupo 40 - FIAP 8IADT Tech Challenge")
    print("="*80 + "\n")
    
    interface.launch(theme=gr.themes.Soft()) # type: ignore
