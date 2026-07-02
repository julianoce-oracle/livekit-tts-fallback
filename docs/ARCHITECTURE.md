# Arquitetura

## Limite de responsabilidade

O LiveKit controla ordem, retries, disponibilidade, recuperação, resampling e a proteção
contra fallback depois do primeiro áudio. A biblioteca fornece providers que traduzem seus
transportes para `livekit.agents.tts.TTS` e um construtor que mantém o lifecycle da cadeia.

```text
AgentSession
  -> ManagedFallbackAdapter (subclasse fina do tts.FallbackAdapter)
       -> OciXaiTTS
            -> AsyncConnectionPool
                 -> WebSocket OCI xAI
       -> fallback escolhido pelo usuário
            -> OciSpeechTTS
                 -> OCI SDK / HTTPS
            -> plugin ElevenLabs
            -> qualquer outro livekit.agents.tts.TTS
```

## OCI xAI

`OciXaiTTS` declara `streaming=True`. Cada `SynthesizeStream` aluga uma conexão do pool,
encaminha eventos `text.delta`, envia `text.done` no flush e converte `audio.delta` base64 em
PCM. A conexão volta ao pool somente depois de uma conclusão saudável.

O pool é local ao event loop e não tenta compartilhar sessões entre processos. Escalabilidade
entre processos deve ser feita criando uma instância por worker e respeitando os limites de
sessão do serviço.

## OCI Speech

`OciSpeechTTS` declara `streaming=False`, pois precisa receber a fala completa antes da
requisição. `synthesize()` executa o SDK síncrono em uma thread, lê a resposta progressivamente
e entrega os bytes ao `AudioEmitter`. O LiveKit decodifica WAV ou MP3 para frames PCM.

Quando a cadeia é usada pela API streaming, o próprio `FallbackAdapter` envolve providers não
streaming com o `StreamAdapter` e segmenta o texto em frases.

## Falha e recuperação

```text
request -> OCI xAI
  -> falhou antes do primeiro frame
  -> LiveKit marca OCI xAI indisponível
  -> LiveKit tenta o próximo provider
  -> tarefa nativa de recuperação testa OCI xAI
  -> recuperação bem-sucedida
  -> OCI xAI volta a ser priorizado nas próximas falas
```

Se qualquer frame já tiver sido emitido, o LiveKit não chama outro provider para a mesma fala.
Esse limite evita repetição audível e mistura de vozes.

## Extensão

A função `build_fallback_tts` recebe instâncias, não nomes de providers. Assim, adicionar um
provider não exige registro global nem alteração no roteador. Uma integração pode ser um plugin
oficial do LiveKit ou uma classe própria derivada de `tts.TTS`.

Informações como `TransportCapabilities` são descritivas. O fallback não toma decisões com
base nelas; o provider continua responsável por pooling, prewarm e encerramento do transporte.
