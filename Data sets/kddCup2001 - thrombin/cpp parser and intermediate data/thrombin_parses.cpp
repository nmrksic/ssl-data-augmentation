#include<stdio.h>
char line[1000000];

int i_array[4000000];
int j_array[4000000];

int count = 0;

int main()
{

FILE *f = fopen("thrombin.data", "r");

// the file has

printf("opened");

char c;
int i = 1;
int j = 0;
char label[2000];



while(i<1910&&!feof(f))
{
    c = fgetc(f);
    label[i] = c;
    printf("%d %c\n",i, c);
    c = fgetc(f);
    j=0;

   while(c!='\n'&&!feof(f)) {

        if(c=='0'||c=='1') {
            j++;

            if(c=='1') {
                //need to save this
                count++;
                i_array[count] = i;
                j_array[count] = j;
               // printf("%d %d\n", i,j);
            }

        }

      c = fgetc(f);

   }

    i++;

}

printf("out of the loop!");

    FILE *f1 = fopen("x_array.dat", "w");
    FILE *f2 = fopen("y_array.dat", "w");
    FILE *f3 = fopen("labels.txt", "w");

    for(int k = 1; k < i; k++) { if(label[k]=='I') fprintf(f3, "%d\n", 0); else fprintf(f3, "%d\n", 1); }

    for(int k = 1; k <= count; k++) { fprintf(f1, "%d ", i_array[k]);   fprintf(f2, "%d ", j_array[k]);   }


}
