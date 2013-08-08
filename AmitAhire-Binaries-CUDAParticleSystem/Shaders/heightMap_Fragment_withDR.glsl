#version 150 core

uniform sampler2D diffuseTex ;
uniform sampler2D bumpTex ; // New !

uniform sampler2D snowTex; // My-New!
uniform sampler2D grassTex; //My-New
uniform sampler2D sandTex; //My-New
uniform sampler2D seaBedTex;

uniform sampler2D bumpGrassTex;
uniform vec3 cameraPos ;
uniform vec4 lightColour ;
uniform vec3 lightPos ;
uniform float lightRadius ;

in Vertex {
	vec3 colour ;
	vec2 texCoord ;
	vec3 normal ;
	vec3 tangent ; // New !
	vec3 binormal ; // New !
	vec3 worldPos ;
} IN;

out vec4 gl_FragColor[2];

void main ( void ) {

	vec4 diffuse = texture ( diffuseTex, IN.texCoord );
	vec4 snow = texture ( snowTex, IN.texCoord );
	vec4 grass = texture ( grassTex, IN.texCoord );
	vec4 sand = texture ( sandTex, IN.texCoord );
	vec4 seaBed = texture( seaBedTex, IN.texCoord);
	
	mat3 TBN = mat3 (IN. tangent , IN. binormal , IN. normal );

	vec3 normal1 = normalize (TBN * ( texture ( bumpGrassTex ,IN. texCoord ). rgb * 2.0 - 1.0));
	vec3 normal2 = normalize (TBN * ( texture ( bumpTex ,IN. texCoord ). rgb * 2.0 - 1.0));
	
	vec3 normal = normal1*normal2;
	
	vec3 incident = normalize ( lightPos - IN. worldPos );
	float lambert = max (0.0 , dot ( incident , normal )); // Different !
 	float dist = length ( lightPos - IN. worldPos );
	float atten = 1.0 - clamp ( dist / lightRadius , 0.0 , 1.0);

 	vec3 viewDir = normalize ( cameraPos - IN. worldPos );
 	vec3 halfDir = normalize ( incident + viewDir );

	float rFactor = max (0.0 , dot ( halfDir , normal )); // Different !
	float sFactor = pow ( rFactor , 33.0 );

	vec3 worldPos_Modifly = IN.worldPos; //Used To Change the world size
	
	//Setup Snow Part
	if(worldPos_Modifly.y > 270.0f)
	{		
		vec3 colour = ( snow .rgb * lightColour .rgb );
		colour += ( lightColour .rgb * sFactor ) * (worldPos_Modifly.y + 200.0f)/1000.0f;
		gl_FragColor[0] = vec4 ( colour * atten * lambert , diffuse .a);
		gl_FragColor[0] .rgb += ( snow .rgb * lightColour .rgb ) * 0.112;
	} 
	//Setup grass Part
	else if(worldPos_Modifly.y >= 200.0f && worldPos_Modifly.y <= 270.0f )
	{		 
		vec3 colour = ( mix(grass.rgb,snow.rgb,1-(270-worldPos_Modifly.y)/70) * lightColour .rgb );
		colour += ( lightColour .rgb * sFactor ) * (worldPos_Modifly.y + 200.0f)/1000.0f;
		gl_FragColor[0] = vec4 ( colour * atten * lambert , diffuse .a);
		gl_FragColor[0] .rgb += ( mix(grass.rgb,snow.rgb,1-(270-worldPos_Modifly.y)/70)  * lightColour .rgb ) * 0.112;
	} 
	else if ((worldPos_Modifly.y >= 130.0f && worldPos_Modifly.y < 230.0f )){		
		vec3 colour = ( grass .rgb * lightColour .rgb );
		colour += ( lightColour .rgb * sFactor ) * (worldPos_Modifly.y + 200.0f)/1000.0f;
		gl_FragColor[0] = vec4 ( colour * atten * lambert , diffuse .a);
		gl_FragColor[0] .rgb += ( grass .rgb * lightColour .rgb ) * 0.112;
	} 
	else if ((worldPos_Modifly.y >= 80.0f && worldPos_Modifly.y < 130.0f )){		
		vec3 colour = ( mix(sand.rgb,grass.rgb,1-(130-worldPos_Modifly.y)/50)  .rgb * lightColour .rgb );
		colour += ( lightColour .rgb * sFactor ) * (worldPos_Modifly.y + 200.0f)/1000.0f;
		gl_FragColor[0] = vec4 ( colour * atten * lambert , diffuse .a);
		gl_FragColor[0] .rgb += ( mix(sand.rgb,grass.rgb,1-(130-worldPos_Modifly.y)/50)  .rgb * lightColour .rgb ) * 0.112;
	}
	//Setup sand Part	
	else if ((worldPos_Modifly.y >= 50.0f && worldPos_Modifly.y < 80.0f )){		
		vec3 colour = ( sand .rgb * lightColour .rgb );
		colour += ( lightColour .rgb * sFactor ) * (worldPos_Modifly.y + 200.0f)/1000.0f;
		gl_FragColor[0] = vec4 ( colour * atten * lambert , diffuse .a);
		gl_FragColor[0] .rgb += ( sand .rgb * lightColour .rgb ) * 0.112;
	}else if ((worldPos_Modifly.y >= 35.0f && worldPos_Modifly.y < 50.0f )){		
		vec3 colour = ( mix(seaBed.rgb,sand.rgb,1-(50-worldPos_Modifly.y)/15)  .rgb * lightColour .rgb );
		colour += ( lightColour .rgb * sFactor ) * (worldPos_Modifly.y + 200.0f)/1000.0f;
		gl_FragColor[0] = vec4 ( colour * atten * lambert , diffuse .a);
		gl_FragColor[0] .rgb += ( mix(seaBed.rgb,sand.rgb,1-(50-worldPos_Modifly.y)/15)  .rgb * lightColour .rgb ) * 0.112;
	}	
	else {
		vec3 colour = ( seaBed .rgb * lightColour .rgb );
		colour += ( lightColour .rgb * sFactor )* (worldPos_Modifly.y + 200.0f)/1000.0f;
		gl_FragColor[0] = vec4 ( colour * atten * lambert , diffuse .a);
		gl_FragColor[0] .rgb += ( seaBed .rgb * lightColour .rgb ) * 0.112;
	}	
	
	gl_FragColor[1] = vec4( normal .xyz * 0.5 + 0.5 ,1.0);
 

	//vec3 BWcolor = vec3((gl_FragColor[0].r+gl_FragColor[0].g+gl_FragColor[0].b)/3.0,(gl_FragColor[0].r+gl_FragColor[0].g+gl_FragColor[0].b)/3.0,(gl_FragColor[0].r+gl_FragColor[0].g+gl_FragColor[0].b)/3.0);
	//gl_FragColor[0].rgb = BWcolor;
	//vec3 Pinkcolor = vec3(gl_FragColor[0].r*1.2,gl_FragColor[0].g*0.2,gl_FragColor[0].b*0.7);

	//vec3 filterColor = vec3(0.0,0.8,0);

	//gl_FragColor[0].rgb = gl_FragColor[0].rgb*filterColor + gl_FragColor[0].rgb;
	
	//gl_FragColor[0].rgb = vec3(1.2-gl_FragColor[0].r,1.2-gl_FragColor[0].g,1.2-gl_FragColor[0].b);
	
	//gl_FragColor[0] = gl_FragColor[0]*vec4(1,0,0,0.1);
	
}