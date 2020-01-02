#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "svm.h"
#include "pfsvm.h"
struct svm_model *model;
struct svm_node *x;
//blk_information*************************
struct svm_model *model_blk;
struct svm_node *x_blk;
//***************************************

int main(int argc, char **argv)
{
    IMAGE *org, *dec, *cls;
    int i, j, k, n, label, success;
    int num_class, side_info;
    double th_list[MAX_CLASS / 2], fvector[NUM_FEATURES], sig_gain = 1.0;
    double offset[MAX_CLASS];
    int cls_hist[MAX_CLASS];
    double sn_before, sn_after;
    static char *orgimg = NULL, *decimg = NULL, *modelfile = NULL, *modimg = NULL;

    /****************************blk information****************************/
    static char *modelfile_blk;
    double th_list_blk[MAX_CLASS / 2], fvector_blk[NUM_FEATURES];
    double offset_blk[MAX_CLASS];
    int cls_hist_blk[MAX_CLASS];
    FILE *TUinfo;
    char tmp[1];
    int xp, yp, w, h;
    int cux = 0, cuy = 0;
    int bx = 0, by = 0;
    int t;
    int blkcorner, blkcorner_x, blkcorner_y, direction;
    /***********************************************************************/

    cpu_time();
    setbuf(stdout, 0);
    for (i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-')
        {
            switch (argv[i][1])
            {
            case 'S':
                sig_gain = atof(argv[++i]);
                break;
            default:
                fprintf(stderr, "Unknown option: %s!\n", argv[i]);
                exit(1);
            }
        }
        else
        {
            if (orgimg == NULL)
            {
                orgimg = argv[i];
            }
            else if (decimg == NULL)
            {
                decimg = argv[i];
            }
            else if (modelfile == NULL)
            {
                modelfile = argv[i];
            }
            else if (modelfile_blk == NULL)
            {
                modelfile_blk = argv[i];
            }
            else
            {
                modimg = argv[i];
            }
        }
    }
    if (modimg == NULL)
    {
        printf("Usage: %s [option] original.pgm decoded.pgm model.svm model_blk.svm modified.pgm\n",
               argv[0]);
        printf("    -S num  Gain factor for sigmoid-like function[%f]\n", sig_gain);
        exit(0);
    }

    org = read_pgm(orgimg);
    dec = read_pgm(decimg);
    cls = alloc_image(org->width, org->height, 255);
    if ((model = svm_load_model(modelfile)) == 0 || (model_blk = svm_load_model(modelfile_blk)) == 0 )
    {
        fprintf(stderr, "can't open model file %s %s\n", modelfile, modelfile_blk);
        exit(1);
    }

    num_class = model->nr_class;
    /*何パーセント正確をはかるためにあるプログラムで実際いらない部分***************************/
    set_thresholds_blk(&org, &dec, 1, num_class, th_list, th_list_blk);
    printf("PSNR = %.2f (dB)\n", sn_before = calc_snr(org, dec));
    printf("# of classes = %d\n", num_class);
    printf("Thresholds in blk = {%.1f", th_list[0]);
    for (k = 1; k < num_class / 2; k++)
    {
        printf(", %.1f", th_list[k]);
    }
    printf("}\n");
    printf("Thresholds blk boundary = {%.1f", th_list_blk[0]);
    for (k = 1; k < num_class / 2; k++)
    {
        printf(", %.1f", th_list_blk[k]);
    }
    printf("}\n");
    printf("Gain factor = %f\n", sig_gain);
    x = Malloc(struct svm_node, NUM_FEATURES + 1);
    x_blk = Malloc(struct svm_node, NUM_FEATURES + 1);
    success = 0;
    for (k = 0; k < num_class; k++)
    {
        offset[k] = 0.0;
        offset_blk[k] = 0.0;
        cls_hist[k] = 0;
        cls_hist_blk[k] = 0;
    }
    /***************************************************************************************/
    TUinfo = fopen("TUinfo.log", "rb");
    while (fscanf(TUinfo, "%s%d%d%d%d", tmp, &xp, &yp, &w, &h) != EOF)
    {
        if (tmp[0] == 'C')
        {
            cux = xp;
            cuy = yp;
        }
        else
        {
            bx = cux + xp;
            by = cuy + yp;
            for (i = by; i < by + h; i++)
            {
                for (j = bx; j < bx + w; j++)
                {
                    if (((j != bx + w - 1) && (j != bx) && (i != by + h - 1) && (i != by)) || (j == 0 && i == 0) || (j == org->width - 1 && i == 0) || (j == 0 && i == org->height - 1) || (j == org->width - 1 && i == org->height - 1) || (j == 0 && i != by + h - 1 && i != by) || (i == 0 && j != bx + w - 1 && j != bx) || (j == org->width - 1 && i != by + h - 1 && i != by) || (i == org->height - 1 && j != bx + w - 1 && j != bx))
                    {
                        //in blk
                        get_fvector(dec, i, j, sig_gain, fvector);
                        n = 0;
                        for (k = 0; k < NUM_FEATURES; k++)
                        {
                            if (fvector[k] != 0.0)
                            {
                                x[n].index = k + 1;
                                x[n].value = fvector[k];
                                n++;
                            }
                        }
                        x[n].index = -1;
                        label = (int)svm_predict(model, x);
                        if (label == get_label(org, dec, i, j, num_class, th_list))
                        {
                            success++;
                        }
                        cls->val[i][j] = label;
                        offset[label] += org->val[i][j] - dec->val[i][j];
                        cls_hist[label]++;
                    }
                    else if ((j == bx + w - 1 && j != org->width - 1) || (j == bx && j != 0) || (i == by + h - 1 && i != org->height - 1) || (i == by && i != 0))
                    {
                        //blk boundary

                        //ブロックの隅であるとき
        								if( (j == bx && i == by && j != 0 && i != 0)
        								||(j == bx && i == by + h - 1 && j != 0 && i != org->height-1)
        								||(j == bx + w - 1 && i == by && j != org->width-1 && i != 0)
        								||(j == bx + w - 1 && i == by + h - 1 && j != org->width-1 && i != org->height-1))
        								{
        									blkcorner_x = j % 4;
        									blkcorner_y = i % 4;
        									switch(blkcorner_x){
        										case 0:
        										if(blkcorner_y == 0){blkcorner = 1;}//左上
        										else{blkcorner = 3;}//左下
        										break;
        										case 3:
        										if(blkcorner_y == 0){blkcorner = 2;}//右上
        										else{blkcorner = 4;}//右下
        										break;
        									}
        									direction = slope(org, i, j, blkcorner)
        								}

        								//境界線方向：horizon上
        								if( (i == by && i != 0 && j != bx && j != bx + w - 1)
        								||(i == by && i != 0 && j == 0)
        								||(i == by && i != 0 && j == org->width-1))
        								{
        									direction = 0;
        								}

        								//境界線方向：horizon下
        								else if( (i == by + h - 1 && i != org->height-1 && j != bx && j != bx + w - 1)//yokosita
        								||(i == by + h - 1 && i != org->height-1 && j == 0)
        								||(i == by + h - 1 && i != org->height-1 && j == org->width-1))
        								{
        									direction = 2;
        								}

        								//境界線方向：vertical右
        								else if(  (j == bx + w - 1 && j != org->width-1 && i != by && i != by + h -1)//tatemigi
        								||(j == bx + w - 1 && j != org->width-1 && i == 0)
        								||(j == bx + w - 1 && j != org->width-1 && i == org->height-1))
        								{
        									direction = 1;
        								}

        								//境界線方向：vertical左
        								else if((j == bx && j != 0 && i != by && i != by + h - 1)//tatehidari
        								||(j == bx && j != 0 && i == 0)
        								||(j == bx && j != 0 && i == org->height-1))
        								{
        									direction = 3;
        								}

                        get_fvector_blk(dec, i, j, sig_gain, fvector_blk, direction);
                        t = 0;
                        for (k = 0; k < NUM_FEATURES; k++)
                        {
                            if (fvector_blk[k] != 0.0)
                            {
                                x_blk[t].index = k + 1;
                                x_blk[t].value = fvector_blk[k];
                                t++;
                            }
                        }
                        x_blk[t].index = -1;
                        label = (int)svm_predict(model_blk, x_blk);
                        if (label == get_label(org, dec, i, j, num_class, th_list_blk))
                        {
                            success++;
                        }
                        cls->val[i][j] = label;
                        offset_blk[label] += org->val[i][j] - dec->val[i][j];
                        cls_hist_blk[label]++;
                    }
                }
            }
        }
        fprintf(stderr, ".");
    }

    fprintf(stderr, "\n");
    fclose(TUinfo);

    printf("Accuracy = %.2f (%%)\n", 100.0 * success / (dec->width * dec->height));
    side_info = 0;
    for (k = 0; k < num_class; k++)
    {
        if (cls_hist[k] > 0)
        {
            offset[k] /= cls_hist[k]; /*offset値は各クラスの再生誤差の平均値*/
        }
        printf("Offset[%d] = %.2f (%d)\n", k, offset[k], cls_hist[k]);
        offset[k] = n = floor(offset[k] + 0.5);
        if (n < 0)
            n = -n;
        side_info += (n + 1); // unary code
        if (n > 0)
            side_info++; // sign bit
    }
    for (k = 0; k < num_class; k++)
    {
        if (cls_hist_blk[k] > 0)
        {
            offset_blk[k] /= cls_hist_blk[k];
        }
        printf("Offset[%d] = %.2f (%d)\n", k, offset_blk[k], cls_hist_blk[k]);
        offset_blk[k] = n = floor(offset_blk[k] + 0.5);
        if (n < 0)
            n = -n;
        side_info += (n + 1);
        if (n > 0)
            side_info++;
    }
    TUinfo = fopen("TUinfo.log", "rb");
    while (fscanf(TUinfo, "%s%d%d%d%d", tmp, &xp, &yp, &w, &h) != EOF)
    {
        if (tmp[0] == 'C')
        {
            cux = xp;
            cuy = yp;
        }
        else
        {
            bx = cux + xp;
            by = cuy + yp;
            for (i = by; i < by + h; i++)
            {
                for (j = bx; j < bx + w; j++)
                {
                    if (((j != bx + w - 1) && (j != bx) && (i != by + h - 1) && (i != by)) || (j == 0 && i == 0) || (j == org->width - 1 && i == 0) || (j == 0 && i == org->height - 1) || (j == org->width - 1 && i == org->height - 1) || (j == 0 && i != by + h - 1 && i != by) || (i == 0 && j != bx + w - 1 && j != bx) || (j == org->width - 1 && i != by + h - 1 && i != by) || (i == org->height - 1 && j != bx + w - 1 && j != bx))
                    {
                        //in blk
                        label = cls->val[i][j];
                        k = dec->val[i][j] + offset[label]; /*オフセット値の加算*/
                        if (k < 0)
                            k = 0;
                        if (k > 255)
                            k = 255;
                        dec->val[i][j] = k;
                    }
                    else if ((j == bx + w - 1 && j != org->width - 1) || (j == bx && j != 0) || (i == by + h - 1 && i != org->height - 1) || (i == by && i != 0))
                    {
                        //blk boundary
                        label = cls->val[i][j];
                        k = dec->val[i][j] + offset_blk[label];
                        if (k < 0)
                            k = 0;
                        if (k > 255)
                            k = 255;
                        dec->val[i][j] = k;
                    }
                }
            }
        }
    }
    printf("PSNR = %.3f (dB)\n", sn_after = calc_snr(org, dec)); /*pfsvmによる輝度補償の時のPSNR*/
    printf("GAIN = %+.3f (dB)\n", sn_after - sn_before);
    printf("SIDE_INFO = %d (bits)\n", side_info);
    write_pgm(dec, modimg);
    svm_free_and_destroy_model(&model);
    svm_free_and_destroy_model(&model_blk);
    free(x);
    free(x_blk);
    printf("cpu time: %.2f sec.\n", cpu_time());
    return (0);
}
