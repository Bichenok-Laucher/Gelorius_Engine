#include "../gelorius.cpp"

void main()
{
   const apart =  Gelorius.createAPart(1, 1024, 1024, texture: "../images/city.jpg");
   apart.show();

   // custom implement apart.implement { ... };
}