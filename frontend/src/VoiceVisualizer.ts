import io from 'socket.io-client';

const socket = io();

export class VoiceVisualizer {
  private logo: HTMLElement;
  private dots: HTMLElement;
  private canvas: HTMLCanvasElement;
  private statusLabel: HTMLElement;
  private canvasCtx: CanvasRenderingContext2D;
  private state: 'idle' | 'listening' | 'processing' | 'speaking';
  private audioContext: AudioContext | null;
  private analyser: AnalyserNode | null;
  private animationFrameId: number | null;
  private mediaRecorder: MediaRecorder | null;
  private audioChunks: Blob[];
  private audioQueue: ArrayBuffer[];
  private isPlaying: boolean;

  constructor(logoEl: HTMLElement, dotsEl: HTMLElement, canvasEl: HTMLCanvasElement, statusLabelEl: HTMLElement) {
    this.logo = logoEl;
    this.dots = dotsEl;
    this.canvas = canvasEl;
    this.statusLabel = statusLabelEl;
    
    if (!this.logo || !this.dots || !this.canvas || !this.statusLabel) {
      throw new Error("VoiceVisualizer: One or more required DOM elements were not provided.");
    }

    this.canvasCtx = this.canvas.getContext('2d')!;
    this.state = 'idle';
    this.audioContext = null;
    this.analyser = null;
    this.animationFrameId = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.audioQueue = [];
    this.isPlaying = false;

    // The component will be responsible for the initial innerHTML
  }

  private _transition(elementToShow: HTMLElement, elementToHide: HTMLElement | null, newLabel: string): Promise<void> {
    return new Promise(resolve => {
      if (elementToHide) {
        elementToHide.classList.add('swirl-out-animation');
        elementToHide.addEventListener('animationend', () => {
          elementToHide.classList.add('hidden');
          elementToHide.classList.remove('swirl-out-animation');
          
          elementToShow.classList.remove('hidden');
          elementToShow.classList.add('swirl-in-animation');
          this.statusLabel.textContent = newLabel;
          elementToShow.addEventListener('animationend', () => {
            elementToShow.classList.remove('swirl-in-animation');
            resolve();
          }, { once: true });

        }, { once: true });
      } else {
        elementToShow.classList.remove('hidden');
        elementToShow.classList.add('swirl-in-animation');
        this.statusLabel.textContent = newLabel;
        elementToShow.addEventListener('animationend', () => {
          elementToShow.classList.remove('swirl-in-animation');
          resolve();
        }, { once: true });
      }
    });
  }

  public startListening() {
    this.state = 'listening';
    this._transition(this.logo, this.dots, 'Listening...');
    this.logo.classList.add('listening');

    this.audioChunks = [];

    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        this.mediaRecorder = new MediaRecorder(stream);

        this.mediaRecorder.ondataavailable = event => {
          if (event.data.size > 0) {
            this.audioChunks.push(event.data);
          }
        };

        this.mediaRecorder.onstop = () => {
          const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
          const reader = new FileReader();
          reader.onloadend = () => {
            const base64data = reader.result;
            socket.emit('transcribe_audio', { audio: base64data });
          };
          reader.readAsDataURL(audioBlob);
        };

        this.mediaRecorder.start();

        setTimeout(() => {
          if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
          }
        }, 4000);
      })
      .catch(err => {
        console.error("Mic error:", err);
        this.stop();
      });
  }

  public startProcessing(transcribedText: string) {
    if (this.state !== 'listening') return;
    this.state = 'processing';
    this.logo.classList.remove('listening');
    this._transition(this.dots, this.logo, 'Processing...').then(() => {
      socket.emit('voice_chat', { text: transcribedText, history: [] }); // Note: history is not available here
    });
  }

  public startSpeaking() {
    if (this.state !== 'processing') return;
    this.state = 'speaking';
    this._transition(this.canvas, this.dots, 'Speaking...').then(() => {
      this.audioQueue = [];
      this.isPlaying = false;
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this.drawWaveform();
    });
  }

  public receiveAudio(audioChunk: ArrayBuffer) {
    if (this.state !== 'speaking') {
      this.startSpeaking();
    }
    this.audioQueue.push(audioChunk);
    if (!this.isPlaying) {
      this.playNextChunk();
    }
  }

  private playNextChunk() {
    if (this.audioQueue.length === 0) {
      this.isPlaying = false;
      return;
    }
    this.isPlaying = true;
    const audioBuffer = this.audioQueue.shift()!;
    this.audioContext!.decodeAudioData(audioBuffer, (buffer) => {
      const source = this.audioContext!.createBufferSource();
      source.buffer = buffer;
      source.connect(this.analyser!);
      this.analyser!.connect(this.audioContext!.destination);
      source.onended = () => this.playNextChunk();
      source.start(0);
    });
  }

  public stop() {
    this.state = 'idle';
    this.logo.classList.remove('listening');
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    this._transition(this.logo, this.dots.classList.contains('hidden') ? this.canvas : this.dots, 'Ready');
  }

  private drawWaveform() {
    this.animationFrameId = requestAnimationFrame(() => this.drawWaveform());

    const width = this.canvas.width;
    const height = this.canvas.height;
    this.canvasCtx.clearRect(0, 0, width, height);
    this.canvasCtx.lineWidth = 2;
    this.canvasCtx.strokeStyle = '#93c5fd';
    this.canvasCtx.beginPath();

    const sliceWidth = width * 1.0 / 128;
    let x = 0;

    const time = Date.now() * 0.01;
    for (let i = 0; i < 128; i++) {
      const v = 0.5 + Math.sin(i * 0.1 + time) * 0.2 + Math.sin(i * 0.05 + time) * 0.2;
      const y = v * height / 2;

      if (i === 0) {
        this.canvasCtx.moveTo(x, y);
      } else {
        this.canvasCtx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    this.canvasCtx.lineTo(width, height / 2);
    this.canvasCtx.stroke();
  }
}
