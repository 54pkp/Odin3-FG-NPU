// SPDX-License-Identifier: GPL-3.0-or-later
// Offline device replay using the exact pinned Kedavra test engine.
#define main kedavra_builtin_main
#include "kfg_vktest.c"
#undef main

#include "fsr4npu.h"
#include <errno.h>
#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

enum { WIDTH = 1280, HEIGHT = 720, PIXELS = WIDTH * HEIGHT };
enum { MODEL_INPUT_PAIR = 2048 };

static void require_ok(int condition, const char *error)
{
	if (!condition) {
		fprintf(stderr, "%s\n", error);
		exit(1);
	}
}

static void pair_path(char *path, size_t cap, const char *folder, int pair, const char *name)
{
	const int used = snprintf(path, cap, "%s/pair-%02d/%s", folder, pair, name);
	require_ok(used >= 0 && (size_t)used < cap, "Replay path is too long");
}

static void frame_io(const char *folder, int pair, const char *name, uint8_t *pixels, size_t bytes, bool write)
{
	char path[1024];
	pair_path(path, sizeof(path), folder, pair, name);
	FILE *file = fopen(path, write ? "wb" : "rb");
	require_ok(file != NULL, path);
	size_t moved = write ? fwrite(pixels, 1, bytes, file) : fread(pixels, 1, bytes, file);
	require_ok(moved == bytes && (write || fgetc(file) == EOF), path);
	require_ok(fclose(file) == 0, path);
}

static void dma_sync(int fd, uint64_t flags)
{
	struct dma_buf_sync sync = { .flags = flags };
	int err;
	do {
		err = ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync);
	} while (err < 0 && errno == EINTR);
	require_ok(err == 0, "DMA-BUF CPU access sync failed");
}

static fsr4npu_stats infer_pair(fsr4npu *daemon, int fd, uint8_t *input, const uint8_t *residual,
				const uint8_t *first, const uint8_t *last, uint8_t *output, bool aligned)
{
	dma_sync(fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE);
	for (size_t i = 0; i < PIXELS; ++i) {
		memcpy(input + 6 * i, first + 4 * i, 3);
		memcpy(input + 6 * i + 3, last + 4 * i, 3);
	}
	dma_sync(fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);

	fsr4npu_stats stats = {0};
	require_ok(fsr4npu_run(daemon, -1, 1000, &stats) == S_OK && stats.exec_ns, "HTP replay failed");
	dma_sync(fd, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
	for (size_t i = 0; i < PIXELS; ++i) {
		const double gate = aligned ? first[i * 4 + 3] / 255.0 : 1.0;
		for (size_t c = 0; c < 3; ++c) {
			const size_t index = i * 3 + c;
			const double mean = ((double)first[i * 4 + c] + last[i * 4 + c]) * 0.5;
			const double correction = ((int)residual[index] - 126) * 0.00384376361034811 * 255.0 * gate;
			output[index] = (uint8_t)lrint(fmin(255, fmax(0, mean + correction)));
		}
	}
	dma_sync(fd, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
	return stats;
}

static double record_measure(engine *e, const kfg_plan *plan)
{
	double timestamps[KFG_MAX_PASSES];
	uint32_t count = 0;
	require_ok(run_plan(e, plan, timestamps, &count) == 0 && count == plan->n,
		   "Kedavra could not execute and time the complete plan");
	double total = 0;
	for (uint32_t i = 0; i < count; ++i)
		total += timestamps[i];
	return total / 1e6;
}

int main(int argc, char **argv)
{
	require_ok(argc == 4 || argc == 5, "usage: kfg-anvil-replay SPIRV_DIR PAIR_DIR COUNT [ANALYSIS_WIDTH]");
	char *tail = NULL;
	long count = strtol(argv[3], &tail, 10);
	require_ok(!*tail && count >= 1 && count <= 32, "COUNT must be from 1 to 32");
	long analysis_width = argc == 5 ? strtol(argv[4], &tail, 10) : 640;
	require_ok(!*tail && analysis_width >= 320 && analysis_width <= WIDTH && analysis_width % 16 == 0,
		   "ANALYSIS_WIDTH must be a multiple of 16 between 320 and 1280");
	kfg_analysis_pixels = analysis_width * (analysis_width * 9 / 16);
	kfg_vk vk;
	require_ok(kfg_vk_init(&vk, argv[1], false) == 0, "Could not initialize the Vulkan replay");
	engine e;
	require_ok(engine_init(&e, &vk, WIDTH, HEIGHT, VK_FORMAT_R8G8B8A8_UNORM) == 0,
		   "Could not allocate Kedavra replay resources");
	kfg_vk_image_destroy(&vk, &e.output);
	require_ok(kfg_vk_image_create(&vk, &e.output, VK_FORMAT_R8G8B8A8_UNORM, 2 * WIDTH, HEIGHT) == 0,
		   "Could not allocate the alignment target");
	const kfg_tuning tune = kfg_default_tuning();
	printf("device=%s driver=%s analysis_width=%ld\n", vk.device_name, vk.driver_info, analysis_width);

	fsr4npu *daemon = NULL;
	fsr4npu_info info = {0};
	require_ok(fsr4npu_connect("fsr4npud_anvil_s_playtest", 100, &daemon, &info) == S_OK &&
		   info.backend == B_QNN && info.ntensors == 2 && info.bytes[0] == 6 * PIXELS &&
		   info.bytes[1] == 3 * PIXELS && !strcmp(info.graph, "anvil_s_float"), "Expected the private W8A8 HTP graph");
	int fd = -1;
	uint64_t total = 0, offsets[FSR4NPU_MAXT] = {0};
	require_ok(fsr4npu_alloc(daemon, 0, &fd, &total, offsets) == S_OK && fd >= 0 &&
		   offsets[0] <= total && info.bytes[0] <= total - offsets[0] &&
		   offsets[1] <= total && info.bytes[1] <= total - offsets[1] &&
		   offsets[0] + info.bytes[0] <= offsets[1], "Invalid shared tensor layout");
	uint8_t *shared = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	require_ok(shared != MAP_FAILED, "Could not map the replay DMA-BUF");
	uint8_t *first = malloc(4 * PIXELS), *last = malloc(4 * PIXELS);
	uint8_t *aligned_first = malloc(4 * PIXELS), *aligned_last = malloc(4 * PIXELS), *output = malloc(3 * PIXELS);
	uint8_t *both = malloc(8 * PIXELS);
	require_ok(first && last && aligned_first && aligned_last && output && both, "Could not allocate replay frames");

	for (int pair = 0; pair < count; ++pair) {
		frame_io(argv[2], pair, "first.rgba", first, 4 * PIXELS, false);
		frame_io(argv[2], pair, "last.rgba", last, 4 * PIXELS, false);
		require_ok(kfg_vk_upload_rgba8(&vk, &e.color[0], first) == 0 &&
			   kfg_vk_upload_rgba8(&vk, &e.color[1], last) == 0, "Could not upload the replay frame pair");
		kfg_plan plan;
		kfg_plan_analyze(&plan, WIDTH, HEIGHT, 1, 0, 0, 0, -1, &tune);
		const double previous = record_measure(&e, &plan);
		kfg_plan_analyze(&plan, WIDTH, HEIGHT, 0, 1, 1, 1, -1, &tune);
		const double motion = record_measure(&e, &plan);
		float scene_stats[7 * 4];
		require_ok(kfg_vk_download_rgba16f(&vk, &e.stats[1], scene_stats) == 0, "Could not read scene-cut evidence");
		kfg_plan_generate(&plan, WIDTH, HEIGHT, 0, 1, 1, 0.5f, MODEL_INPUT_PAIR);
		const double warp = record_measure(&e, &plan);
		require_ok(kfg_vk_download_rgba8(&vk, &e.output, both) == 0, "Could not read the aligned pair");
		for (size_t row = 0; row < HEIGHT; ++row) {
			memcpy(aligned_first + row * 4 * WIDTH, both + row * 8 * WIDTH, 4 * WIDTH);
			memcpy(aligned_last + row * 4 * WIDTH, both + (2 * row + 1) * 4 * WIDTH, 4 * WIDTH);
		}
		frame_io(argv[2], pair, "aligned-first.rgba", aligned_first, 4 * PIXELS, true);
		frame_io(argv[2], pair, "aligned-last.rgba", aligned_last, 4 * PIXELS, true);
		const fsr4npu_stats raw = infer_pair(daemon, fd, shared + offsets[0], shared + offsets[1],
						   first, last, output, false);
		frame_io(argv[2], pair, "unaligned-htp.rgb", output, 3 * PIXELS, true);
		const fsr4npu_stats aligned = infer_pair(daemon, fd, shared + offsets[0], shared + offsets[1],
						       aligned_first, aligned_last, output, true);
		frame_io(argv[2], pair, "aligned-htp.rgb", output, 3 * PIXELS, true);
		printf("pair=%02d previous=%.4f motion=%.4f warp=%.4f raw_htp=%.4f aligned_htp=%.4f "
		       "cut=%.0f frac_bad=%.5f compensated_error=%.5f direct_error=%.5f\n",
		       pair, previous, motion, warp, raw.exec_ns / 1e6, aligned.exec_ns / 1e6,
		       scene_stats[0], scene_stats[1], scene_stats[2], scene_stats[3]);
		fflush(stdout);
	}
	free(first);
	free(last);
	free(aligned_first);
	free(aligned_last);
	free(output);
	free(both);
	munmap(shared, total);
	close(fd);
	fsr4npu_close(daemon);
	engine_destroy(&e);
	kfg_vk_destroy(&vk);
	return 0;
}
