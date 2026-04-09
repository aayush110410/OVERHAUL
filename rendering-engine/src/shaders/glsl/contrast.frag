// Contrast + brightness adjustment
uniform sampler2D tDiffuse;
uniform float contrast;
uniform float brightness;
varying vec2 vUv;

void main() {
  vec4 texel = texture2D(tDiffuse, vUv);

  // Apply brightness
  vec3 color = texel.rgb + vec3(brightness);

  // Apply contrast around midpoint 0.5
  color = (color - 0.5) * contrast + 0.5;

  // Clamp
  color = clamp(color, 0.0, 1.0);

  gl_FragColor = vec4(color, texel.a);
}
