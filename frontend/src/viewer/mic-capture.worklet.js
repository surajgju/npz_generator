class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const size = options?.processorOptions?.bufferSize;
    this.bufferSize = Number.isFinite(size) && size > 0 ? size : 2048;
    this.buffer = new Float32Array(this.bufferSize);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs && inputs[0] && inputs[0][0];
    if (!input) return true;
    for (let i = 0; i < input.length; i++) {
      this.buffer[this.offset++] = input[i];
      if (this.offset >= this.bufferSize) {
        // Post a copy so the main thread owns its own buffer slice.
        this.port.postMessage(this.buffer.slice());
        this.offset = 0;
        this.buffer = new Float32Array(this.bufferSize);
      }
    }
    return true;
  }
}

registerProcessor("mic-capture-processor", MicCaptureProcessor);
