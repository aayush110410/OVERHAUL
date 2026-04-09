// Bloom extraction + composite fragment shader
uniform sampler2D tDiffuse;
uniform float bloomStrength;
uniform float bloomRadius;
uniform float bloomThreshold;
varying vec2 vUv;

// 13-tap Gaussian blur kernel
vec4 blur13(sampler2D tex, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.411764705882353)  * direction / resolution;
  vec2 off2 = vec2(3.2941176470588234) * direction / resolution;
  vec2 off3 = vec2(5.176470588235294)  * direction / resolution;
  color += texture2D(tex, uv) * 0.1964825501511404;
  color += texture2D(tex, uv + off1) * 0.2969069646728344;
  color += texture2D(tex, uv - off1) * 0.2969069646728344;
  color += texture2D(tex, uv + off2) * 0.09447039785044732;
  color += texture2D(tex, uv - off2) * 0.09447039785044732;
  color += texture2D(tex, uv + off3) * 0.010381362401148057;
  color += texture2D(tex, uv - off3) * 0.010381362401148057;
  return color;
}

void main() {
  vec4 texel = texture2D(tDiffuse, vUv);

  // Extract bright areas
  float brightness = dot(texel.rgb, vec3(0.2126, 0.7152, 0.0722));
  vec3 bloom = vec3(0.0);
  if (brightness > bloomThreshold) {
    bloom = texel.rgb * (brightness - bloomThreshold) * bloomStrength;
  }

  // Simple radial blur approximation
  vec2 center = vec2(0.5);
  vec2 dir = (vUv - center) * bloomRadius;
  vec4 blurred = blur13(tDiffuse, vUv, vec2(1024.0), dir);

  vec3 result = texel.rgb + bloom + blurred.rgb * bloomStrength * 0.3;
  gl_FragColor = vec4(result, texel.a);
}
