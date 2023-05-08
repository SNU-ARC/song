#include <stdio.h>
#include <string.h>
#include "logger.h"
#include "parser_dense.h"
#include "parser.h"
#include "data.h"
#include "kernelgraph.h"
#include "config.h"
#include <chrono>

std::unique_ptr<Data> data;
std::unique_ptr<GraphWrapper> graph; 
int topk = 0;
int display_topk = 1;

void build_callback(idx_t idx,std::vector<std::pair<int,value_t>> point){
	if (idx % 10000 == 0)
		printf("idx: %d, point_size: %d\n", idx, point.size());
	data->add(idx,point);
	graph->add_vertex(idx,point);
}

std::vector<std::vector<std::pair<int,value_t>>> batch_queries;
std::vector<std::vector<idx_t>> results(ACC_BATCH_SIZE); // 1000000
std::vector<std::vector<idx_t>> groundtruths(ACC_BATCH_SIZE); // 1000000
unsigned int groundtruth_dim = 0;

void flush_queries(int set_single_batch) {
	unsigned int topk_hit = 0;

	int old_batch_size = batch_queries.size();
	int new_batch_size = old_batch_size;
	int new_num_iter = 1;
	
	if (set_single_batch) {
		new_batch_size = 1;
		new_num_iter = old_batch_size;
	}

	fprintf(stderr, "Original Batch size : %d, New batch size : %d, Num iter : %d\n", old_batch_size, new_batch_size, new_num_iter);

	for (int q = 0; q < new_num_iter; q++) {
		std::vector<std::vector<std::pair<int, value_t>>> batch_queries_size_new;

		for (int i = 0; i < new_batch_size; i++) {
			batch_queries_size_new.push_back(batch_queries[q * new_batch_size + i]);
		}

		fprintf(stderr, "Batch size after insert query : %d\n", batch_queries_size_new.size());

		std::vector<std::vector<idx_t>> results_size_new(new_batch_size);

		graph->search_top_k_batch(batch_queries_size_new, topk, results_size_new);

		for (int i = 0; i < new_batch_size; ++i) {
			std::vector<idx_t> result;
			std::vector<idx_t> groundtruth;

			result = results_size_new[i];
			groundtruth = groundtruths[q * new_batch_size + i];

			for (int j = 0; j < display_topk; j++) {
				for (int k = 0; k < display_topk; k++) {
					if (result[j] == groundtruth[k]) {
						topk_hit++;
						break;
					}
				}
			}
		}
	}
	fprintf(stderr, "Recall@%d: %lf\%\n", display_topk, (float)topk_hit / (new_batch_size * new_num_iter * display_topk) * 100);

	WarpAStarAccelerator::free_all();
	batch_queries.clear();
}

void query_callback(idx_t idx,std::vector<std::pair<int,value_t>> point){
	batch_queries.push_back(point);
	// Uncomment the following lines to have a finer granularity batch processing
	//if(batch_queries.size() == ACC_BATCH_SIZE){
	//    flush_queries();
	//}
	/////////////////////
}

void load_groundtruth(const char* groundtruth_path) {
	auto fp = fopen(groundtruth_path, "rb");
	if (fp == NULL) {
		Logger::log(Logger::ERROR, "File not found at (%s)\n", groundtruth_path);
		exit(1);
	}
	fread(&groundtruth_dim, sizeof(unsigned int), 1, fp);

	fseek(fp, 0, SEEK_END);
	size_t fsize = ftell(fp);
	unsigned int num = (unsigned int)(fsize / (groundtruth_dim + 1) / 4);

	fseek(fp, 0 ,SEEK_SET);
	for (size_t i = 0; i < num; i++) {
		unsigned int value; 
		fseek(fp, 4, SEEK_CUR);
		for (unsigned int j = 0; j < groundtruth_dim; j++) {
			fread(&value, sizeof(unsigned int), 1, fp);
			groundtruths[i].push_back((idx_t)value);
		}
	}
	fclose(fp);
}

void usage(char** argv){
	printf("Usage: %s <build/test> <build_data> <query_data> <search_top_k> <row> <dim> <return_top_k> <l2/ip/cos> <groundtruth>\n",argv[0]);
}

value_t* WarpAStarAccelerator::d_data = NULL;
value_t* WarpAStarAccelerator::d_query = NULL;
idx_t* WarpAStarAccelerator::d_result = NULL;
idx_t* WarpAStarAccelerator::d_graph = NULL;

int main(int argc,char** argv){
	if(argc != 10 && argc != 11){
		usage(argv);
		return 1;
	}
	// You may need to increase this parameter for some new GPUs
	cudaDeviceSetLimit(cudaLimitMallocHeapSize,800*1024*1024);
	//////////////////////
	size_t row = atoll(argv[5]);
	int dim = atoi(argv[6]);

	display_topk = atoi(argv[7]);
	std::string dist_type = argv[8];
	data = std::unique_ptr<Data>(new Data(row,dim));
	if(dist_type == "l2"){
		graph = std::unique_ptr<GraphWrapper>(new KernelFixedDegreeGraph<0>(data.get())); 
	}else if(dist_type == "ip"){
		graph = std::unique_ptr<GraphWrapper>(new KernelFixedDegreeGraph<1>(data.get())); 
	}else if(dist_type == "cos"){
		graph = std::unique_ptr<GraphWrapper>(new KernelFixedDegreeGraph<2>(data.get())); 
	}else{
		usage(argv);
		return 1;
	}
	std::string mode = std::string(argv[1]);
	topk = atoi(argv[4]);
	if(mode == "build"){
		//std::unique_ptr<ParserDense> build_parser(new ParserDense(argv[2],build_callback));
		std::unique_ptr<Parser> build_parser(new Parser(argv[2], build_callback));
		fprintf(stderr,"Writing the graph and data...");    
		data->dump();
		fprintf(stderr,"...");    
		graph->dump();
		fprintf(stderr,"done\n");    
	}else if(mode == "test"){
		fprintf(stderr,"Loading the graph and data...");    
		data->load();
		fprintf(stderr,"...");    
		graph->load();
		fprintf(stderr,"done\n");    
		load_groundtruth(argv[9]);
		fprintf(stderr,"Loading groundtruth done\n");
		//std::unique_ptr<ParserDense> query_parser(new ParserDense(argv[3],query_callback));
		std::unique_ptr<Parser> query_parser(new Parser(argv[3],query_callback));
		int set_single_batch = atoi(argv[10]);
		flush_queries(set_single_batch);
	}else{
		usage(argv);
		return 1;
	}
	return 0;
}
