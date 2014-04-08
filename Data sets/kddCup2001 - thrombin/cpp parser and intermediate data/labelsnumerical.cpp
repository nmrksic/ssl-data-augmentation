#include<stdio.h>

int main()
{

FILE *f = fopen("ThrombinKey.txt", "r");
FILE *g = fopen("ThrombinKeyNum.txt","w");
// the file has
char c;
while(!feof(f))
{
    c = fgetc(f);
    if(c=='I') fprintf(g, "%d\n", 0);
    if(c=='A') fprintf(g, "%d\n", 1);

}

}
