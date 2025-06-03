#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/types.h>
#include "filter.h"
#include "signal.h"
#include "timing.h"

#define MAXWIDTH 40
#define THRESHOLD 2.0
#define ALIENS_LOW  50000.0
#define ALIENS_HIGH 150000.0


typedef struct {
    int thread_id;
    int num_threads;
    int num_processors;
    int filter_order;
    int num_bands;
    double Fs;
    double* signal_data;
    int num_samples;
    double* band_power;
} thread_args_t;

void* thread_fn(void* arg) {
    thread_args_t* targs = (thread_args_t*) arg;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(targs->thread_id % targs->num_processors, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    double Fc = targs->Fs / 2.0;
    double bandwidth = Fc / targs->num_bands;
    double filter_coeffs[targs->filter_order + 1];

    for (int band = targs->thread_id; band < targs->num_bands; band += targs->num_threads) {
        generate_band_pass(targs->Fs,band * bandwidth + 0.0001,(band + 1) * bandwidth - 0.0001,targs->filter_order,filter_coeffs);
        hamming_window(targs->filter_order, filter_coeffs);
        convolve_and_compute_power(targs->num_samples,targs->signal_data,targs->filter_order,filter_coeffs,&targs->band_power[band]);
    }

    return NULL;
}
void usage() {
  printf("usage: p_band_scan text|bin|mmap signal_file Fs filter_order num_bands num_threads num_processors\n");
}


double avg_power(double* data, int num) {

  double ss = 0;
  for (int i = 0; i < num; i++) {
    ss += data[i] * data[i];
  }

  return ss / num;
}

double max_of(double* data, int num) {

  double m = data[0];
  for (int i = 1; i < num; i++) {
    if (data[i] > m) {
      m = data[i];
    }
  }
  return m;
}

double avg_of(double* data, int num) {

  double s = 0;
  for (int i = 0; i < num; i++) {
    s += data[i];
  }
  return s / num;
}

void remove_dc(double* data, int num) {

  double dc = avg_of(data,num);

  printf("Removing DC component of %lf\n",dc);

  for (int i = 0; i < num; i++) {
    data[i] -= dc;
  }
}



int main(int argc, char* argv[]) {

    if (argc != 8) {
        usage();
        return -1;
    }

    char sig_type    = toupper(argv[1][0]);
    char* sig_file   = argv[2];
    double Fs        = atof(argv[3]);
    int filter_order = atoi(argv[4]);
    int num_bands    = atoi(argv[5]);
    int num_threads  = atoi(argv[6]);
    int num_procs    = atoi(argv[7]);

    assert(Fs > 0.0);
    assert(filter_order > 0 && !(filter_order & 1));
    assert(num_bands > 0);
    assert(num_threads > 0);
    assert(num_procs > 0);

    printf("type:     %s\nfile:     %s\nFs:       %lf Hz\norder:    %d\nbands:    %d\nthreads:  %d\nprocessors: %d\n",sig_type == 'T' ? "Text" : (sig_type == 'B' ? "Binary" : (sig_type == 'M' ? "Mapped Binary" : "UNKNOWN TYPE")),
      sig_file, Fs, filter_order, num_bands, num_threads, num_procs);

    printf("Load or map file\n");

    signal* sig;
    switch (sig_type) {
        case 'T': sig = load_text_format_signal(sig_file); break;
        case 'B': sig = load_binary_format_signal(sig_file); break;
        case 'M': sig = map_binary_format_signal(sig_file); break;
        default:
            printf("Unknown signal type\n");
            return -1;
    }

    if (!sig) {
        printf("Unable to load or map file\n");
        return -1;
    }

    sig->Fs = Fs;
    remove_dc(sig->data, sig->num_samples);

    double* band_power = calloc(num_bands, sizeof(double));
    pthread_t threads[num_threads];
    thread_args_t args[num_threads];

    for (int i = 0; i < num_threads; i++) {
        args[i] = (thread_args_t){
            .thread_id = i,
            .num_threads = num_threads,
            .num_processors = num_procs,
            .filter_order = filter_order,
            .num_bands = num_bands,
            .Fs = Fs,
            .signal_data = sig->data,
            .num_samples = sig->num_samples,
            .band_power = band_power
        };
        pthread_create(&threads[i], NULL, thread_fn, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    double Fc = Fs / 2.0;
    double bandwidth = Fc / num_bands;
    double max_band_power = max_of(band_power, num_bands);
    double avg_band_power = avg_of(band_power, num_bands);

    double lb = -1, ub = -1;
    int wow = 0;

    for (int band = 0; band < num_bands; band++) {
        double band_low  = band * bandwidth + 0.0001;
        double band_high = (band + 1) * bandwidth - 0.0001;

        printf("%5d %20lf to %20lf Hz: %20lf ", band, band_low, band_high, band_power[band]);
        for (int i = 0; i < MAXWIDTH * (band_power[band] / max_band_power); i++) {
            printf("*");
        }

        if ((band_low >= ALIENS_LOW && band_low <= ALIENS_HIGH) ||
            (band_high >= ALIENS_LOW && band_high <= ALIENS_HIGH)) {

            if (band_power[band] > THRESHOLD * avg_band_power) {
                printf("(WOW)");
                wow = 1;
                if (lb < 0) lb = band_low;
                ub = band_high;
            } else {
                printf("(meh)");
            }
        } else {
            printf("(meh)");
        }

        printf("\n");
    }

    if (wow)
        printf("POSSIBLE ALIENS %lf-%lf HZ (CENTER %lf HZ)\n", lb, ub, (lb + ub) / 2.0);
    else
        printf("no aliens\n");

    free(band_power);
    free_signal(sig);
    return 0;
}
