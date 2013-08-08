 #version 150 core
 uniform sampler2D diffuseTex ;
 uniform sampler2D emissiveTex ;
 uniform sampler2D specularTex ;
uniform sampler2D damageTex ;

 uniform float clock;

 in Vertex {
	vec2 texCoord ;
 } IN;

 out vec4 gl_FragColor ;

 void main ( void ) {
	 vec3 diffuse = texture ( diffuseTex , IN. texCoord ). xyz ;	 
	 vec3 light = texture ( emissiveTex , IN. texCoord ). xyz ;
	 vec3 specular = texture ( specularTex , IN. texCoord ). xyz ;
	 vec3 damage = texture(damageTex,IN. texCoord ). xyz ;
	 
	 light = light * vec3(10);
	 
	gl_FragColor .xyz = diffuse * ((13.5-abs(clock-12))/12); // ambient
	gl_FragColor .xyz += diffuse * light ; // lambert
	gl_FragColor .xyz += specular*0.5 ; // Specular
	//gl_FragColor .xyz += damage*1 ; // Specular
	gl_FragColor .a = 1.0;

	//gl_FragColor.xyz  = damage;
	//gl_FragColor.rgb = diffuse;
	//gl_FragColor.rgb = light;
	//gl_FragColor.rgb = light+specular;
	//gl_FragColor.rgb = diffuse+light;	 
 }