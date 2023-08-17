## Configurando a GPU para uso de redes neurais com PyTorch e Tensorflow

### PyTorch
PyTorch é um pacote de machine-learning baseada no pacote torch e ganhou ampla adocão na comunidade de deep-learning devido ao seu gráfico computacional dinâmico e suporte para aceleracão de GPU.

### O que é CUDA?
CUDA é uma plataforma de computacão paralela e modelo de programacão desenvolvida pela NVIDIA. Permite o uso das GPUs NVIDIA para tarefas de computacão de uso geral, incluindo aprendizado profundo.

A principio o uso de GPUs pelo PyTorch é compatível somente com suporte de placas de vídeo NVIDIA devido ao uso do CUDA e CUDnn.

Em MacOs a aceleracão só é permitida com usa de GPUs AMD ou Apple Silicon com o MPS (Metal Performance Shaders) - Consultar: 
  - [Accelerated Training - Hugging Face](https://huggingface.co/docs/accelerate/usage_guides/mps)
  - [Accelerated Pytorch Training on Mac](https://developer.apple.com/metal/pytorch/)

### Windows
Para a versão Windows com CUDA 11.7 :

#### Passo 1 -- Configuracão das Variáveis de ambiente e instalacão de CUDnn e CUDA:
Faca download do CUDnn para a versão específica do CUDA escolhidos no link que foi selecionado os parâmetros de compatibilidade do sistema

1- [Download CUDA](https://developer.nvidia.com/cuda-11-6-0-download-archive)

Os arquivos baixados do CUDnn deverá ser extraidos para:

Copy bin\cudnn*.dll para C:\Program Files\NVIDIA\CUDNN\v8.x\bin.

Copy include\cudnn*.h para C:\Program Files\NVIDIA\CUDNN\v8.x\include.

Copy lib\cudnn*.lib para C:\Program Files\NVIDIA\CUDNN\v8.x\lib.

2- [Download CUDnn](https://developer.nvidia.com/rdp/cudnn-download)



#### Passo 2 -- Configurando as variáveis de ambiente no Windows

Procure no menu executar "Editar as variáveis de ambiente do sistema" e na aba Avançado procure o botão "Variáveis de ambiente..."

Editar variáveis de ambiente do sistema >> Avançado >> Variáveis de ambiente...

- Adicione às **VARIÁVEIS DE USUÁRIO** os caminhos nas variáveis **Path**:
  
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin

  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\extras\CUPTI\lib64

  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\include

  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\libnvvp

  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x

![Alt img](/.readme/alt_img.png)

- Crie uma **VARIÁVEL DE USUÁRIO** e nas **VARIÁVEIS DO SISTEMA** chamada **CUDNN** e adicione os mesmos caminhos anteriormente adicionados na variável **Path:**

![Alt img2](/.readme/alt_img2.png)

- Adicione também os endereços a seguir na variável Path de **VARIÁVEL DO SISTEMA:**

  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\libnvvp

  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin

![Alt img2]/.readme/(alt_img3.png)

#### Passo 3 -- Instale a versão correta 

Supomos assim que iremos instalar o PyTorch com a versão de CUDA 11.7:

**Utilize este link para selecionar os parâmetros de compatibilidade do seu sistema LOCAL**
  - [Get Started Locally](https://pytorch.org/get-started/locally/)

Copie o comando impresso na aba **"Run this command"** e utilize para instalar o PyTorch:
> DESINSTALE QUALQUER OUTRA VERSÃO O PYTORCH INSTALADO ANTERIORMENTE PARA EVITAR PROBLEMAS DE CONFLITO ENTRE VERSÕES.

![Alt text](/.readme/image-1.png)

```python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

Dessa forma verifique se está disponível para o uso!!

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

