// Fog depth — linear-depth fog blending
uniform sampler2D tDiffuse;
uniform sampler2D tDepth;
uniform float fogNear;
uniform float fogFar;
uniform vec3 fogColor;
uniform float cameraNear;
uniform float cameraFar;
varying vec2 vUv;

float linearizeDepth(float d) {
  float z = d * 2.0 - 1.0; // NDC
  return (2.0 * cameraNear * cameraFar) / (cameraFar + cameraNear - z * (cameraFar - cameraNear));
}

void main() {
  vec4 texel = texture2D(tDiffuse, vUv);
  float depth = texture2D(tDepth, vUv).r;
  float linearDepth = linearizeDepth(depth);

  float fogFactor = smoothstep(fogNear, fogFar, linearDepth);
  vec3 result = mix(texel.rgb, fogColor, fogFactor);

  gl_FragColor = vec4(result, texel.a);
}
