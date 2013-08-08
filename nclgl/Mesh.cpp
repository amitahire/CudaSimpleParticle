#include "Mesh.h"

//Constructor-Destructor
Mesh :: Mesh ( void ) {
	for( int i = 0; i < MAX_BUFFER ; ++i) {
		bufferObject [i] = 0;
	}
	//Unique name identifier VAO
	glGenVertexArrays (1, &arrayObject);
	numVertices = 0;
	vertices = NULL;
	colours = NULL;
	type = GL_TRIANGLES;

	//Texture init
	texture = 0;
	textureCoords = NULL;

	//Indice init
	indices = NULL;
	numIndices = 0;

	//Normal init
	normal = NULL;

	//Bump Map init
	tangents = NULL ;
	bumpTexture = 0;
}

Mesh ::~ Mesh ( void ) {
	glDeleteVertexArrays (1, &arrayObject);
	//number of buffer, in bufferObject
	glDeleteBuffers (MAX_BUFFER,bufferObject);
	glDeleteTextures(1,&texture);
	//delete Bump map

		

	glDeleteTextures (1 ,& bumpTexture );

	//Clear Array of data
	delete [] tangents;
	delete [] textureCoords;
	delete [] vertices;
	delete [] colours;
	delete [] indices;
	delete [] normal;
}

Mesh *Mesh::GenerateTriangle() {
	//Allocate New Mesh
	Mesh *m = new Mesh ();
	m-> numVertices = 3;
	
	//Setup Vertices
	m-> vertices = new Vector3 [m-> numVertices];
	m-> vertices [0] = Vector3 (0.0f, 0.5f, 0.0f);
	m-> vertices [1] = Vector3 (0.5f, -0.5f, 0.0f);
	m-> vertices [2] = Vector3 (-0.5f, -0.5f, 0.0f);

	//Setup Colours
	m-> colours = new Vector4 [m-> numVertices];
	m-> colours [0] = Vector4 (1.0f, 0.0f, 0.0f ,0.0f);
	m-> colours [1] = Vector4 (0.0f, 1.0f, 0.0f ,0.0f);
	m-> colours [2] = Vector4 (0.0f, 0.0f, 1.0f ,0.0f);

	//Setup Texture
	m-> textureCoords = new Vector2 [m-> numVertices];
	m-> textureCoords [0] = Vector2 (0.5f , 0.0f );
	m-> textureCoords [1] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [2] = Vector2 (0.0f , 1.0f );

	//Buffer Data into current mesh then return pointer to object
	m-> BufferData ();
	return m;
}

Mesh * Mesh::GenerateQuad() {
		Mesh * m = new Mesh ();
		m -> numVertices = 4;
		m -> type = GL_TRIANGLE_STRIP ;

		m -> vertices = new Vector3 [m -> numVertices ];
		m -> textureCoords = new Vector2 [m -> numVertices ];
		m -> colours = new Vector4 [m -> numVertices ];
		m -> normal = new Vector3 [m -> numVertices ];
		m -> tangents = new Vector3 [m -> numVertices ];

		m -> vertices [0] = Vector3 ( -1.0f , -1.0f , 0.0f );
		m -> vertices [1] = Vector3 ( -1.0f , 1.0f , 0.0f );
		m -> vertices [2] = Vector3 (1.0f , -1.0f , 0.0f );
		m -> vertices [3] = Vector3 (1.0f , 1.0f , 0.0f );

		m -> textureCoords [0] = Vector2 (0.0f , 1.0f );
		m -> textureCoords [1] = Vector2 (0.0f , 0.0f );
		m -> textureCoords [2] = Vector2 (1.0f , 1.0f );
		m -> textureCoords [3] = Vector2 (1.0f , 0.0f );

	for (int i = 0; i < 4; ++ i ) {
		m -> colours [ i ] = Vector4 (1.0f , 1.0f ,1.0f ,1.0f );
		m -> normal [ i ] = Vector3 (0.0f , 0.0f , -1.0f );
		m -> tangents [ i ] = Vector3 (1.0f , 0.0f ,0.0f );
	}

	m -> BufferData ();

return m ;
}

Mesh *Mesh::GenerateBox() {
	Mesh *m = new Mesh ();
	m->type =  GL_TRIANGLES;
	m-> numVertices = 36;

	m-> vertices = new Vector3 [m-> numVertices ];	
		
	m-> vertices [0] = Vector3 (1.0f, 1.0f, -1.0f);
	m-> vertices [1] = Vector3 (1.0f, -1.0f, -1.0f);
	m-> vertices [2] = Vector3 (-1.0f, -1.0f, -1.0f);	
	m-> vertices [3] = Vector3 (-1.0f, -1.0f, -1.0f);
	m-> vertices [4] = Vector3 (1.0f, 1.0f, -1.0f);
	m-> vertices [5] = Vector3 (-1.0f, 1.0f, -1.0f);

	m-> vertices [6] = Vector3 (1.0f, 1.0f, 1.0f);	
	m-> vertices [7] = Vector3 (1.0f, -1.0f, 1.0f);
	m-> vertices [8] = Vector3 (-1.0f, -1.0f, 1.0f);
	m-> vertices [9] = Vector3 (-1.0f, -1.0f, 1.0f);
	m-> vertices [10] = Vector3 (1.0f, 1.0f, 1.0f);	
	m-> vertices [11] = Vector3 (-1.0f, 1.0f, 1.0f);
	
	m-> vertices [12] = Vector3 (1.0f, -1.0f, 1.0f);
	m-> vertices [13] = Vector3 (1.0f, 1.0f, 1.0f);
	m-> vertices [14] = Vector3 (1.0f, -1.0f, -1.0f);
	m-> vertices [15] = Vector3 (1.0f, 1.0f, 1.0f);
	m-> vertices [16] = Vector3 (1.0f, -1.0f, -1.0f);
	m-> vertices [17] = Vector3 (1.0f, 1.0f, -1.0f);
	
	m-> vertices [18] = Vector3 (-1.0f, -1.0f, 1.0f);	
	m-> vertices [19] = Vector3 (-1.0f, 1.0f, 1.0f);
	m-> vertices [20] = Vector3 (-1.0f, -1.0f, -1.0f);
	m-> vertices [21] = Vector3 (-1.0f, -1.0f, -1.0f);
	m-> vertices [22] = Vector3 (-1.0f, 1.0f, 1.0f);	
	m-> vertices [23] = Vector3 (-1.0f, 1.0f, -1.0f);

	m-> vertices [24] = Vector3 (-1.0f, 1.0f, 1.0f);
	m-> vertices [25] = Vector3 (1.0f, 1.0f, 1.0f);
	m-> vertices [26] = Vector3 (1.0f, 1.0f, -1.0f);	
	m-> vertices [27] = Vector3 (1.0f, 1.0f, -1.0f);
	m-> vertices [28] = Vector3 (-1.0f, 1.0f, 1.0f);
	m-> vertices [29] = Vector3 (-1.0f, 1.0f, -1.0f);

	m-> vertices [30] = Vector3 (1.0f, -1.0f, 1.0f);	
	m-> vertices [31] = Vector3 (1.0f, -1.0f, -1.0f);
	m-> vertices [32] = Vector3 (-1.0f, -1.0f, -1.0f);
	m-> vertices [33] = Vector3 (-1.0f, -1.0f, -1.0f);
	m-> vertices [34] = Vector3 (1.0f, -1.0f, 1.0f);	
	m-> vertices [35] = Vector3 (-1.0f, -1.0f, 1.0f);
		
	m-> colours = new Vector4 [m-> numVertices ];
	
	/*for( int i = 0; i < 6 ; i++){
		m-> colours [i] = Vector4 (0.0f, 1.0f, 0.0f ,0.4f);	
		m-> colours [i+6] = Vector4 (1.0f, 0.5f, 0.0f ,0.4f);	
		m-> colours [i+12] = Vector4 (1.0f, 0.0f, 0.0f ,0.4f);	
		m-> colours [i+18] = Vector4 (1.0f, 1.0f, 0.0f ,0.4f);	
		m-> colours [i+24] = Vector4 (0.0f, 0.0f, 1.0f ,0.4f);	
		m-> colours [i+30] = Vector4 (1.0f, 0.0f, 1.0f ,0.4f);	
	}*/

	for(int i = 0; i < 36;i++){
		m-> colours [i] = Vector4 (0.0f, 0.0f, 0.0f ,1.0f);	
	}
	
	m-> textureCoords = new Vector2 [m-> numVertices ];
	m-> textureCoords [0] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [1] = Vector2 (0.0f , 1.0f );
	m-> textureCoords [2] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [3] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [4] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [5] = Vector2 (1.0f , 0.0f );

	m-> textureCoords [6] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [7] = Vector2 (0.0f , 1.0f );
	m-> textureCoords [8] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [9] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [10] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [11] = Vector2 (1.0f , 0.0f );
	
	m-> textureCoords [12] = Vector2 (1.0f , 0.0f );
	m-> textureCoords [13] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [14] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [15] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [16] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [17] = Vector2 (0.0f , 1.0f );
	
	m-> textureCoords [18] = Vector2 (0.0f , 1.0f );
	m-> textureCoords [19] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [20] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [21] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [22] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [23] = Vector2 (1.0f , 0.0f );
	
	m-> textureCoords [24] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [25] = Vector2 (0.0f , 1.0f );
	m-> textureCoords [26] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [27] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [28] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [29] = Vector2 (1.0f , 0.0f );
	
	m-> textureCoords [30] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [31] = Vector2 (1.0f , 0.0f );
	m-> textureCoords [32] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [33] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [34] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [35] = Vector2 (0.0f , 1.0f );
	
	m-> BufferData ();
	return m;
}

Mesh *Mesh::GenerateM(float r,float g,float b,float a){

	Mesh *m = new Mesh();
	m->type =  GL_TRIANGLES;
	m->numVertices = 36;
	
	m-> vertices = new Vector3 [m-> numVertices];	
	m-> vertices [0] = Vector3 (5.0f/5.0f, -2.5f/5.0f, 0.0f);
	m-> vertices [1] = Vector3 (4.0f/5.0f, -2.5f/5.0f, 0.0f);
	m-> vertices [2] = Vector3 (2.5f/5.0f, 3.5f/5.0f, 0.0f);	

	m-> vertices [3] = Vector3 (2.5f/5.0f, 3.5f/5.0f, 0.0f);
	m-> vertices [4] = Vector3 (4.0f/5.0f, -2.5f/5.0f, 0.0f);
	m-> vertices [5] = Vector3 (2.5f/5.0f, 1.5f/5.0f, 0.0f);
	
	m-> vertices [6] = Vector3 (2.5f/5.0f, 1.5f/5.0f, 0.0f);
	m-> vertices [7] = Vector3 (2.5f/5.0f, 3.5f/5.0f, 0.0f);
	m-> vertices [8] = Vector3 (2.0f/5.0f, 0.5f/5.0f, 0.0f);	

	m-> vertices [9] = Vector3 (2.0f/5.0f, 0.5f/5.0f, 0.0f);
	m-> vertices [10] = Vector3 (2.5f/5.0f, 3.5f/5.0f, 0.0f);
	m-> vertices [11] = Vector3 (1.0f/5.0f, 0.5f/5.0f, 0.0f);

	m-> vertices [12] = Vector3 (1.5f/5.0f, -0.5f/5.0f, 0.0f);
	m-> vertices [13] = Vector3 (0.5f/5.0f, -0.5f/5.0f, 0.0f);
	m-> vertices [14] = Vector3 (0.0f/5.0f, -3.5f/5.0f, 0.0f);

	m-> vertices [15] = Vector3 (0.5f/5.0f, -0.5f/5.0f, 0.0f);	
	m-> vertices [16] = Vector3 (0.0f/5.0f, -1.5f/5.0f, 0.0f);
	m-> vertices [17] = Vector3 (0.0f/5.0f, -3.5f/5.0f, 0.0f);

	m-> vertices [18] = Vector3 (-5.0f/5.0f, -2.5f/5.0f, 0.0f);
	m-> vertices [19] = Vector3 (-4.0f/5.0f, -2.5f/5.0f, 0.0f);
	m-> vertices [20] = Vector3 (-2.5f/5.0f, 3.5f/5.0f, 0.0f);	

	m-> vertices [21] = Vector3 (-2.5f/5.0f, 3.5f/5.0f, 0.0f);
	m-> vertices [22] = Vector3 (-4.0f/5.0f, -2.5f/5.0f, 0.0f);
	m-> vertices [23] = Vector3 (-2.5f/5.0f, 1.5f/5.0f, 0.0f);
	
	m-> vertices [24] = Vector3 (-2.5f/5.0f, 1.5f/5.0f, 0.0f);
	m-> vertices [25] = Vector3 (-2.5f/5.0f, 3.5f/5.0f, 0.0f);
	m-> vertices [26] = Vector3 (-2.0f/5.0f, 0.5f/5.0f, 0.0f);	

	m-> vertices [27] = Vector3 (-2.0f/5.0f, 0.5f/5.0f, 0.0f);
	m-> vertices [28] = Vector3 (-2.5f/5.0f, 3.5f/5.0f, 0.0f);
	m-> vertices [29] = Vector3 (-1.0f/5.0f, 0.5f/5.0f, 0.0f);

	m-> vertices [30] = Vector3 (-1.5f/5.0f, -0.5f/5.0f, 0.0f);
	m-> vertices [31] = Vector3 (-0.5f/5.0f, -0.5f/5.0f, 0.0f);
	m-> vertices [32] = Vector3 (-0.0f/5.0f, -3.5f/5.0f, 0.0f);

	m-> vertices [33] = Vector3 (-0.5f/5.0f, -0.5f/5.0f, 0.0f);	
	m-> vertices [34] = Vector3 (-0.0f/5.0f, -1.5f/5.0f, 0.0f);
	m-> vertices [35] = Vector3 (-0.0f/5.0f, -3.5f/5.0f, 0.0f);

	m-> colours = new Vector4 [m-> numVertices ];
	for( int i = 0; i < 36 ; i++){
		m-> colours [i] = Vector4 (r,g,b,a);	
	}
	
	m-> textureCoords = new Vector2 [m-> numVertices ];
	m-> textureCoords [0] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [1] = Vector2 (0.0f , 1.0f );
	m-> textureCoords [2] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [3] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [4] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [5] = Vector2 (1.0f , 0.0f );

	m-> textureCoords [6] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [7] = Vector2 (0.0f , 1.0f );
	m-> textureCoords [8] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [9] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [10] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [11] = Vector2 (1.0f , 0.0f );
	
	m-> textureCoords [12] = Vector2 (1.0f , 0.0f );
	m-> textureCoords [13] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [14] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [15] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [16] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [17] = Vector2 (0.0f , 1.0f );
	
	m-> textureCoords [18] = Vector2 (0.0f , 1.0f );
	m-> textureCoords [19] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [20] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [21] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [22] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [23] = Vector2 (1.0f , 0.0f );
	
	m-> textureCoords [24] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [25] = Vector2 (0.0f , 1.0f );
	m-> textureCoords [26] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [27] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [28] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [29] = Vector2 (1.0f , 0.0f );
	
	m-> textureCoords [30] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [31] = Vector2 (1.0f , 0.0f );
	m-> textureCoords [32] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [33] = Vector2 (1.0f , 1.0f );
	m-> textureCoords [34] = Vector2 (0.0f , 0.0f );
	m-> textureCoords [35] = Vector2 (0.0f , 1.0f );

	m-> BufferData ();
	return m;
}

//Calculate Normal

void Mesh::GenerateNormals() {
	if (! normal ) {
		normal = new Vector3 [ numVertices ];
	}
	for ( GLuint i = 0; i < numVertices ; ++ i ){
		normal[ i ] = Vector3 ();
	}
	if( indices ) { // Generate per - vertex normals
		for ( GLuint i = 0; i < numIndices ; i +=3){
			unsigned int a = indices [ i ];
			unsigned int b = indices [ i +1];
			unsigned int c = indices [ i +2];
			Vector3 n = Vector3 :: Cross (( vertices [ b ] - vertices [ a ]) ,( vertices [ c ] - vertices [ a ]));

			normal[a] += n ;
			normal[b] += n ;
			normal[c] += n ;
		}
	} else { // It 's just a list of triangles , so generate face normals
		for ( GLuint i = 0; i < numVertices ; i +=3){
			Vector3 & a = vertices [ i ];
			Vector3 & b = vertices [ i +1];
			Vector3 & c = vertices [ i +2];

			Vector3 n = Vector3 :: Cross (b -a ,c - a );
			normal [ i ]   = n;
			normal [ i +1] = n;
			normal [ i +2] = n;
		}
	}
	for ( GLuint i = 0; i < numVertices ; ++ i ){
		normal [ i ]. Normalise ();
	}
}

void Mesh::GenerateTangents () {
	if (! textureCoords ) {
		return;
	}
	if (! tangents ) {
		tangents = new Vector3 [ numVertices ];
	}
	for ( GLuint i = 0; i < numVertices ; ++i){
		tangents [i] = Vector3 ();
	}
	if( indices ) {
		for ( GLuint i = 0; i < numIndices ; i +=3){
			int a = indices [i];
			int b = indices [i +1];
			int c = indices [i +2];

			Vector3 tangent = GenerateTangent ( vertices [a], vertices [b],
			vertices [c], textureCoords [a],
			textureCoords [b], textureCoords [c]);

			tangents [a] += tangent ;
			tangents [b] += tangent ;
			tangents [c] += tangent ;
		}
	}
	else {
		for ( GLuint i = 0; i < numVertices ; i +=3){
			Vector3 tangent = GenerateTangent ( vertices [i], vertices [i+1] ,
			vertices [i+2] , textureCoords [i],
			textureCoords [i+1] , textureCoords [i+2]);

			tangents [i] += tangent ;
			tangents [i+1] += tangent ;
			tangents [i+2] += tangent ;
		}
	}
	for ( GLuint i = 0; i < numVertices ; ++i){
		tangents [i]. Normalise ();
	}
}

Vector3 Mesh::GenerateTangent ( const Vector3 &a, const Vector3 &b,const Vector3 &c, const Vector2 &ta ,const Vector2 &tb , const Vector2 &tc) {
	Vector2 coord1 = tb -ta;
	Vector2 coord2 = tc -ta;

	Vector3 vertex1 = b-a;
	Vector3 vertex2 = c-a;

	Vector3 axis = Vector3 ( vertex1 * coord2 .y - vertex2 * coord1 .y);

	float factor = 1.0f / ( coord1 .x * coord2 .y - coord2 .x * coord1 .y);

	return axis * factor ;
}

void Mesh::BufferData(){
	glBindVertexArray (arrayObject); //Buffer Data to this. object
	glGenBuffers(1,&bufferObject[VERTEX_BUFFER]); //Create one buffer to store Vertex data.
	glBindBuffer(GL_ARRAY_BUFFER,bufferObject[VERTEX_BUFFER]); //Use the vertex buffer now, as an array buffer
	glBufferData(GL_ARRAY_BUFFER,numVertices*sizeof(Vector3),vertices,GL_STATIC_DRAW); //Setup the buffer with the size to buffer and the pointer to the data, also with the way to store data (const,dynamic)
	glVertexAttribPointer(VERTEX_BUFFER,3,GL_FLOAT,GL_FALSE,0,0); //Way for OpenGL to access data, (vertex buffer - 3x GL_float per vertex)
	glEnableVertexAttribArray(VERTEX_BUFFER);
	if(textureCoords){ //Buffer Texture Data into Texture Buffer
		glGenBuffers(1,&bufferObject[TEXTURE_BUFFER]);
		glBindBuffer(GL_ARRAY_BUFFER,bufferObject[TEXTURE_BUFFER]);
		glBufferData(GL_ARRAY_BUFFER,numVertices*sizeof (Vector2),textureCoords,GL_STATIC_DRAW);
		glVertexAttribPointer(TEXTURE_BUFFER,2, GL_FLOAT,GL_FALSE,0 ,0);
		glEnableVertexAttribArray(TEXTURE_BUFFER);
	}	
	if(colours){ //Buffer Colour Data into Colour Buffer
		glGenBuffers(1,&bufferObject[COLOUR_BUFFER]);
		glBindBuffer(GL_ARRAY_BUFFER,bufferObject[COLOUR_BUFFER]);
		glBufferData(GL_ARRAY_BUFFER,numVertices*sizeof(Vector4),colours,GL_STATIC_DRAW);
		glVertexAttribPointer(COLOUR_BUFFER,4,GL_FLOAT,GL_FALSE,0,0);
		glEnableVertexAttribArray(COLOUR_BUFFER);
	}
	if(indices){
		glGenBuffers(1,&bufferObject[INDEX_BUFFER]);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,bufferObject[INDEX_BUFFER]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER,numIndices*sizeof(GLuint),indices,GL_STATIC_DRAW);
	}
	if(normal) {
		glGenBuffers (1 , & bufferObject [ NORMAL_BUFFER ]);
		glBindBuffer ( GL_ARRAY_BUFFER , bufferObject [ NORMAL_BUFFER ]);
		glBufferData ( GL_ARRAY_BUFFER , numVertices * sizeof ( Vector3 ) ,normal , GL_STATIC_DRAW );
		glVertexAttribPointer ( NORMAL_BUFFER , 3 , GL_FLOAT , GL_FALSE ,0 ,0);
		glEnableVertexAttribArray ( NORMAL_BUFFER );
	}
	if(tangents){
		glGenBuffers(1,&bufferObject[TANGENT_BUFFER]);
		glBindBuffer(GL_ARRAY_BUFFER,bufferObject[TANGENT_BUFFER]);
		glBufferData(GL_ARRAY_BUFFER,numVertices*sizeof(Vector3),tangents,GL_STATIC_DRAW);
		glVertexAttribPointer(TANGENT_BUFFER,3,GL_FLOAT,GL_FALSE,0,0);
		glEnableVertexAttribArray(TANGENT_BUFFER);
	}
	glBindVertexArray(0); //Done Buffering, end bind
}

void Mesh::Draw(){
	//Bind Texture and object VAO to draw
	glActiveTexture ( GL_TEXTURE0 );
	glBindTexture(GL_TEXTURE_2D,texture);
	//Bind Bump Map Texture
	glActiveTexture ( GL_TEXTURE1 ); // New !!!
	glBindTexture ( GL_TEXTURE_2D , bumpTexture );
	glBindVertexArray (arrayObject);
	//Check if Mesh is in index form or not.
	if(bufferObject[INDEX_BUFFER]){
		glDrawElements(type,numIndices,GL_UNSIGNED_INT,0);
	} else {
		//Draw object specifling the type (GL_TRIANGLES), with starting index and the number of vertices to draw
		glDrawArrays(type,0,numVertices);
	}
	//Ending drawing, end binding
	glBindVertexArray(0);	
	glBindTexture(GL_TEXTURE_2D,0);
}