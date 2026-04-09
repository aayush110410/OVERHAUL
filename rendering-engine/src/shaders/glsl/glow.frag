// Ambient glow — soft radial vignette + additive colour
uniform sampler2D tDiffuse;
uniform float glowIntensity;
uniform vec3 glowColor;
varying vec2 vUv;

void main() {
  vec4 texel = texture2D(tDiffuse, vUv);

  // Radial distance from centre
  vec2 center = vec2(0.5);
  float dist = length(vUv - center);

  // Inverse vignette — brighter at edges is "glow"
  float glow = smoothstep(0.2, 0.9, dist) * glowIntensity;

  vec3 result = texel.rgb + glowColor * glow;
  gl_FragColor = vec4(clamp(result, 0.0, 1.0), texel.a);
}
