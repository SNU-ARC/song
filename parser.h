#pragma once

#include"config.h"
#include"logger.h"
#include<stdlib.h>
#include<memory>
#include<vector>
#include<functional>



class Parser{
private:
	const int ONE_BASED_LIBSVM = 1;
    const int MAX_LINE = 10000000;
    std::function<void(idx_t,std::vector<std::pair<int,value_t>>)> consume;

    std::vector<int> tokenize(char* buff){
        std::vector<int> ret;
        int i = 0;
        while(*(buff + i) != '\0'){
            if(*(buff + i) == ':' || *(buff + i) == ' ')
                ret.push_back(i);
            ++i;
        }
//        for (unsigned m = 0; m < ret.size(); m++)
//          printf("%d ", ret[i]);
//        printf("\n");
        return ret;
    }

    std::vector<std::pair<int,value_t>> parse(std::vector<int> tokens,char* buff){
        std::vector<std::pair<int,value_t>> ret;
        ret.reserve(tokens.size() / 2);
        for(int i = 0;i + 1 < tokens.size();i += 2){
            int index;
            value_t val;
            sscanf(buff + tokens[i] + 1,"%d",&index);
			index -= ONE_BASED_LIBSVM;
			double tmp;
            sscanf(buff + tokens[i + 1] + 1,"%lf",&tmp);
			val = tmp;
            ret.push_back(std::make_pair(index,val));
        }
        return ret;
    }


public:

    Parser(const char* path,std::function<void(idx_t,std::vector<std::pair<int,value_t>>)> consume) : consume(consume){
        auto fp = fopen(path,"r");
        if(fp == NULL){
            Logger::log(Logger::ERROR,"File not found at (%s)\n",path);
            exit(1);
        }

        unsigned int dim;
        fread(&dim, sizeof(unsigned int), 1, fp);

        fseek(fp, 0, SEEK_END);
        size_t fsize = ftell(fp);
        unsigned int num = (unsigned)(fsize / (dim + 1) / 4);

        fseek(fp, 0, SEEK_SET);
        for (size_t i = 0; i < num; i++) {
          fseek(fp, 4, SEEK_CUR);
          std::vector<std::pair<int, value_t> > values; 
          for (size_t j = 0; j < dim; j++) {
            value_t val;
            fread(&val, sizeof(value_t), 1, fp);
            values.push_back(std::make_pair((int)j, val));
          }
          consume ((idx_t)i, values);
//            for (unsigned m = 0; m < values.size(); m++) {
//              printf("(%d, %lf) ", values[m].first, values[m].second);
//            }
//            printf("\n\n");
        }
//        std::unique_ptr<char[]> buff(new char[MAX_LINE]);
//        std::vector<std::string> buffers;
//        idx_t idx = 0;
//        while(fgets(buff.get(),MAX_LINE,fp)){
//            auto tokens = tokenize(buff.get());
//            auto values = parse(tokens,buff.get());
//            consume(idx,values);
////            for (unsigned m = 0; m < tokens.size(); m++) {
////              printf("%d ", tokens[m]);
////            }
////            printf("\n");
//            for (unsigned m = 0; m < values.size(); m++) {
//              printf("(%d, %lf) ", values[m].first, values[m].second);
//            }
//            printf("\n\n");
//            ++idx;
//        }
        fclose(fp);
    }
    
};
