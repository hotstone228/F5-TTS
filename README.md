# F5-TTS Fork

This is a fork of [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS), a Diffusion Transformer-based Text-to-Speech system.

## Russian Model

Russian model available at: [https://huggingface.co/hotstone228/F5-TTS-Russian](https://huggingface.co/hotstone228/F5-TTS-Russian)

## Installation

See installation instructions in the [original repository](https://github.com/SWivid/F5-TTS).

## Usage

### Gradio App

```bash
f5-tts_infer-gradio
```

### CLI Inference

```bash
f5-tts_infer-cli --model "F5-TTS" --ref_audio "ref_audio.wav" --gen_text "Your text here"
```

## License

MIT License for code. Pre-trained models may have additional restrictions.

## Credits

Original repo: [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS)
Russian model: [hotstone228/F5-TTS-Russian](https://huggingface.co/hotstone228/F5-TTS-Russian)
