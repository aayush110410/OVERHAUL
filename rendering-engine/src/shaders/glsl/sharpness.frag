// Sharpness (unsharp-mask) enhancement
uniform sampler2D tDiffuse;
uniform vec2 resolution;
uniform float amount;
varying vec2 vUv;

void main() {
  vec2 texelSize = 1.0 / resolution;
  vec4 center = texture2D(tDiffuse, vUv);

  // Sample neighbours
  vec4 top    = texture2D(tDiffuse, vUv + vec2(0.0,  texelSize.y));
  vec4 bottom = texture2D(tDiffuse, vUv + vec2(0.0, -texelSize.y));
  vec4 left   = texture2D(tDiffuse, vUv + vec2(-texelSize.x, 0.0));
  vec4 right  = texture2D(tDiffuse, vUv + vec2( texelSize.x, 0.0));

  // Laplacian-based sharpen
  vec4 sharp = center * 5.0 - top - bottom - left - right;
  vec4 result = center + sharp * amount;

  gl_FragColor = vec4(clamp(result.rgb, 0.0, 1.0), center.a);
}
