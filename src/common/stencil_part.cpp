#include <stdio.h>
#include <vector>
#include <map>
#include "param.h"

using namespace std;

typedef struct
{
    vector<int> type;//横为0，竖为1
    vector<int> pos;//相对中心点stencil_core的位置，横着：上负下正，竖着：左负右正
    vector<int> order;//只有重新组合的时候会用到，记录当前的行是第几个行，当前的竖是第几个竖，用于快速确定对应matrix的offset
}StencilPart;

ll complexity(StencilPart stencil_part,int block_m,int block_n)//计算复杂度
{
    ll ret = 0;
    ll row_val = block_m*block_n*block_n;
    ll col_val = block_m*block_m*block_n;
    int i;
    for(i = 0; i < stencil_part.type.size(); i++)
    {
        if(stencil_part.type[i] == 0)
            ret += row_val;
        else
            ret += col_val;
    }
    return ret;
}

void show_stencil_part(float *S_h, StencilPart stencil_part,char *s)
{
    int i,j;
    float stencil[stencil_shape_M][stencil_shape_N];
    for(i = 0; i < stencil_shape_M; i++)
        for(j = 0; j < stencil_shape_N; j++)
            stencil[i][j] = S_h[i*stencil_shape_N + j];

    printf("总共拆分为 %d 个部分\n",(int)stencil_part.type.size());
    for(i = 0; i < stencil_part.type.size(); i++)
    {
        if(stencil_part.type[i] == 0)
        {
            printf("横向 %d\n",stencil_part.pos[i] + stencil_core_M);
        }
    }
    for(i = 0; i < stencil_part.type.size(); i++)
    {
        if(stencil_part.type[i] == 1)
        {
            printf("纵向 %d\n",stencil_part.pos[i] + stencil_core_N);
        }
    }

    printf("\n原始Stencil：\n");

    for(i = 0; i < stencil_shape_M; i++)
    {
        for(j = 0; j < stencil_shape_N; j++)
        {
            printf("%6.2f ",stencil[i][j]);
        }
        printf("\n");
    }

    char str[stencil_shape_M][stencil_shape_N];
    for(i = 0; i < stencil_shape_M; i++)
    {
        for(j = 0; j < stencil_shape_N; j++)
        {
            if(iszero(stencil[i][j]))
            {
                str[i][j] = '.';
            }
            else
            {
                str[i][j] = 'a';
            }
        }
    }

    for(i = 0; i < stencil_shape_M; i++)
    {
        for(j = 0; j < stencil_shape_N; j++)
        {
            printf("  %c",str[i][j]);
        }
        printf("\n");
    }

    for(i = 0; i < stencil_part.type.size(); i++)
    {
        if(stencil_part.type[i] == 0)
        {
            for(j = 0; j < stencil_shape_N; j++)
            {
                str[stencil_part.pos[i] + stencil_core_M][j] = 'x';
            }
        }
        else
        {
            for(j = 0; j < stencil_shape_M; j++)
            {
                str[j][stencil_part.pos[i] + stencil_core_N] = 'x';
            }
        }
    }
    printf("\n拆分Stencil：\n");
    for(i = 0; i < stencil_shape_M; i++)
    {
        for(j = 0; j < stencil_shape_N; j++)
        {
            printf("  %c",str[i][j]);
        }
        printf("\n");
    }

    printf("\n");
    printf("拆分方式：%s\n",s);
    
}

StencilPart type_split(float *S_h, int block_m,int block_n,int type)//1为行，2为列
{
    int i,j;
    float stencil[stencil_shape_M][stencil_shape_N];
    for(i = 0; i < stencil_shape_M; i++)
        for(j = 0; j < stencil_shape_N; j++)
            stencil[i][j] = S_h[i*stencil_shape_N + j];

    int row[stencil_shape_M][stencil_shape_N] = {0};//某个位置右侧是否有非0值
    int col[stencil_shape_M][stencil_shape_N] = {0};//某个位置下侧是否有非0值

    for(i = stencil_shape_M - 1; i >= 0; --i)
    {
        for(j = stencil_shape_N - 1; j >= 0; --j)
        {
            if(!iszero(stencil[i][j]))
            {
                row[i][j] = 1;
                col[i][j] = 1;
            }
            if(i < stencil_shape_M - 1)
            {
                col[i][j] += col[i + 1][j];
            }
            if(j < stencil_shape_N - 1)
            {
                row[i][j] += row[i][j + 1];
            }
        }
    }

    StencilPart stencil_part;
    if(type == 1)//行
    {
        for(i = 0; i < stencil_shape_M; i++)
        {
            if(row[i][0] > 0)
            {
                stencil_part.type.push_back(0);
                stencil_part.pos.push_back(i - stencil_core_M);
            }
        }
    }
    else//列
    {
        for(i = 0; i < stencil_shape_N; i++)
        {
            if(col[0][i] > 0)
            {
                stencil_part.type.push_back(1);
                stencil_part.pos.push_back(i - stencil_core_N);
            }
        }
    }
    return stencil_part;
 }

void brute_force(float *S_h, int block_m,int block_n,StencilPart *list,int *num)
{
    int i,j;

    float stencil[stencil_shape_M][stencil_shape_N];
    for(i = 0; i < stencil_shape_M; i++)
        for(j = 0; j < stencil_shape_N; j++)
            stencil[i][j] = S_h[i*stencil_shape_N + j];

    int row[stencil_shape_M][stencil_shape_N] = {0};//某个位置右侧是否有非0值
    int col[stencil_shape_M][stencil_shape_N] = {0};//某个位置下侧是否有非0值

    for(i = stencil_shape_M - 1; i >= 0; --i)
    {
        for(j = stencil_shape_N - 1; j >= 0; --j)
        {
            if(!iszero(stencil[i][j]))
            {
                row[i][j] = 1;
                col[i][j] = 1;
            }
            if(i < stencil_shape_M - 1)
            {
                col[i][j] += col[i + 1][j];
            }
            if(j < stencil_shape_N - 1)
            {
                row[i][j] += row[i][j + 1];
            }
        }
    }

    ll row_major_complexity = stencil_shape_N,col_major_complexity = stencil_shape_M;//搜索复杂度 反过来的
    int row_num = 0,col_num = 0;
    vector<int> row_list;//记录哪些行或者列有非0值
    vector<int> col_list;
    for(i = 0; i < stencil_shape_M; ++i)
    {
        if(row[i][0] > 0)
        {
            row_major_complexity *= 2;
            row_list.push_back(i);
            row_num++;
        }
    }
    for(i = 0; i < stencil_shape_N; ++i)
    {
        if(col[0][i] > 0)
        {
            col_major_complexity *= 2;
            col_list.push_back(i);
            col_num++;
        }
    }

    if(Min(row_major_complexity,col_major_complexity) > 1e7)//搜索复杂度太高就不暴力算了
    {
        return;
    }

    StencilPart best_stencil_part;
    if(row_major_complexity <= col_major_complexity)//行搜索复杂度低一些
    {
        ll max_mask = (1 << row_num) - 1;
        // ll max_mask = (((1 << (row_num - 1)) - 1) << 1) | 1;
        int check[64] = {0};
        for(ll mask = 0; mask <= max_mask; ++mask)
        {
            StencilPart temp;
            memset(check,0,sizeof(check));
            for(i = 0; i < row_num; ++i)
            {
                if(mask & (1 << i))
                {
                    temp.type.push_back(0);
                    temp.pos.push_back(row_list[i] - stencil_core_M);
                }
                else
                {
                    for(j = 0; j < stencil_shape_N; ++j)
                    {
                        if(!iszero(stencil[row_list[i]][j]))
                        {
                            check[j] = 1;
                        }
                    }
                }
            }
            for(i = 0; i < stencil_shape_N; ++i)
            {
                if(check[i] == 1)
                {
                    temp.type.push_back(1);
                    temp.pos.push_back(i - stencil_core_N);
                }
            }

            if(mask == 0 || complexity(temp,block_m,block_n) < complexity(best_stencil_part,block_m,block_n))
            {
                best_stencil_part = temp;
            }
        }

    }
    else
    {
        ll max_mask = (1 << col_num) - 1;
        int check[64] = {0};

        for(ll mask = 0; mask <= max_mask; ++mask)
        {
            StencilPart temp;
            memset(check,0,sizeof(check));
            for(i = 0; i < col_num; ++i)
            {
                if(mask & (1 << i))
                {
                    temp.type.push_back(1);
                    temp.pos.push_back(col_list[i] - stencil_core_N);
                }
                else
                {
                    for(j = 0; j < stencil_shape_M; ++j)
                    {
                        if(!iszero(stencil[j][col_list[i]]))
                        {
                            check[j] = 1;
                        }
                    }
                }
            }
            for(i = 0; i < stencil_shape_M; ++i)
            {
                if(check[i] == 1)
                {
                    temp.type.push_back(0);
                    temp.pos.push_back(i - stencil_core_M);
                }
            }

            if(mask == 0 || complexity(temp,block_m,block_n) < complexity(best_stencil_part,block_m,block_n))
            {
                best_stencil_part = temp;
            }
        }
    }
    
    list[(*num)++] = best_stencil_part;
}

StencilPart greedy(float *S_h,int block_m,int block_n,int weight)
{
    int i,j;

    float stencil[stencil_shape_M][stencil_shape_N];
    for(i = 0; i < stencil_shape_M; i++)
        for(j = 0; j < stencil_shape_N; j++)
            stencil[i][j] = S_h[i*stencil_shape_N + j];

    int row[stencil_shape_M][stencil_shape_N] = {0};//某个位置右侧是否有非0值
    int col[stencil_shape_M][stencil_shape_N] = {0};//某个位置下侧是否有非0值

    for(i = stencil_shape_M - 1; i >= 0; --i)
    {
        for(j = stencil_shape_N - 1; j >= 0; --j)
        {
            if(!iszero(stencil[i][j]))
            {
                row[i][j] = 1;
                col[i][j] = 1;
            }
            if(i < stencil_shape_M - 1)
            {
                col[i][j] += col[i + 1][j];
            }
            if(j < stencil_shape_N - 1)
            {
                row[i][j] += row[i][j + 1];
            }
        }
    }

    StencilPart best;
    int check_row[stencil_shape_M] = {0};
    int check_col[stencil_shape_N] = {0};
    int sum = stencil_shape_M + stencil_shape_N;

    // ll row_val = block_m*block_n*block_n;
    // ll col_val = block_m*block_m*block_n;

    ll row_weight = 1;
    ll col_weight = 1;
    if(weight == 1)
    {
        row_weight = block_m;//看起来写反了，其实没有，因为上面是计算量，而这个权值其实是优先度
        col_weight = block_n;//优先度大的反而计算量小
    }


    while(sum > 0)
    {
        int max_val = 0;
        int isrow = 1;
        int pos = 0;

        for(i = 0; i < stencil_shape_M; i++)
        {
            if(check_row[i] == 0 && row[i][0]*row_weight > max_val)
            {
                max_val = row[i][0]*row_weight;
                pos = i;
                isrow = 1;
            }
        }
        for(i = 0; i < stencil_shape_N; i++)
        {
            if(check_col[i] == 0 && col[0][i]*col_weight > max_val)
            {
                max_val = col[0][i]*col_weight;
                pos = i;
                isrow = 0;
            }
        }

        if(isrow == 1)
        {
            best.type.push_back(0);
            best.pos.push_back(pos - stencil_core_M);
            sum -= 1;
            check_row[pos] = 1;
            for(i = 0; i < stencil_shape_N; i++)
            {
                if(!iszero(stencil[pos][i]))
                {
                    col[0][i] -=1;
                    if(col[0][i] == 0)
                    {
                        sum -= 1;
                        check_col[i] = 1;
                    }
                }
            }
        }
        else
        {
            best.type.push_back(1);
            best.pos.push_back(pos - stencil_core_N);
            sum -= 1;
            check_col[pos] = 1;
            for(i = 0; i < stencil_shape_M; i++)
            {
                if(!iszero(stencil[i][pos]))
                {
                    row[i][0] -=1;
                    if(row[i][0] == 0)
                    {
                        sum -= 1;
                        check_row[i] = 1;
                    }
                }
            }
        }
    }

    return best;
}

StencilPart find_best_stencil_part(float *S_h, int block_m,int block_n)
{
    int num = 0;
    StencilPart list[20];
    char str[][50] = {
        "Greedy算法",
        "Greedy算法（加权）",
        "Brute Force算法"
    };

    list[num++] = greedy(S_h,block_m,block_n,0);
    list[num++] = greedy(S_h,block_m,block_n,1);
    //为了测试sharedM的读取所以注释掉
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    brute_force(S_h,block_m,block_n,list,&num);//因为不一定执行
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ll min_cost = complexity(list[0],block_m,block_n);
    int index = 0;
    for(int i = 1; i < num; i++)
    {
        ll cost = complexity(list[i],block_m,block_n);
        if(cost <= min_cost)
        {
            min_cost = cost;
            index = i;
        }
    }
    show_stencil_part(S_h,list[index],str[index]);

    return list[index];
}

StencilPart find_best_stencil_part(float *S_h, int block_m,int block_n,int type)
{
    char str[][50] = {
        "行拆分",
        "列拆分"
    };
    StencilPart stencil_part;
    if(type == 1)
        stencil_part = type_split(S_h,block_m,block_n,type);//1 为行
    else if(type == 2)
        stencil_part = type_split(S_h,block_m,block_n,type);//2 为列
    else
        return find_best_stencil_part(S_h,block_m,block_n);
    show_stencil_part(S_h,stencil_part,str[type - 1]);
    return stencil_part;
}