"""Genetic-Algorithm partition optimization (2D bin packing variation).

This module solves a fixed-size 2D partition/bin packing problem:
- Given rectangles (boxes) of various sizes.
- Each partition has a fixed (width, height).
- Goal: minimize the number of bins used, while packing them tightly.

GA representation (chromosome): a permutation of partitions with a
rotation 90 bit per partition.
Decoding: First-Fit packing into partitions using a max-rects style
placement heuristic.
Fitness: rewards fewer partitions and higher utilization.
"""

from email.mime import image

import cv2
from networkx import center
import numpy as np
from typing import Dict, Tuple, List, Iterable, Optional, Sequence
from src.DataCleanModel.utils.ai_labeler_json_writer import ShapeInfo
import random
from dataclasses import dataclass
from src.DataCleanModel.partition_merge.partition import Partition
from copy import deepcopy

random.seed(40)
    

@dataclass(frozen=True)
class PlacedPartitiotn:
	partition_id: int
	x: int
	y: int
	width: int
	height: int
	rotated: bool

	@property
	def area(self) -> int:
		return self.width * self.height


@dataclass(frozen=True)
class Rect:
	x: int
	y: int
	width: int
	height: int

	@property
	def right(self) -> int:
		return self.x + self.width

	@property
	def top(self) -> int:
		return self.y + self.height

	def contains(self, other: "Rect") -> bool:
		return (
			other.x >= self.x
			and other.y >= self.y
			and other.right <= self.right
			and other.top <= self.top
		)

	def intersects(self, other: "Rect") -> bool:
		return not (
			other.x >= self.right
			or other.right <= self.x
			or other.y >= self.top
			or other.top <= self.y
		)


class MaxRectsBin:
	"""A single fixed-size bin using a simplified MaxRects free-space model.

	Placement heuristic: Bottom-Left (min y, then min x), with a small tie-breaker
	on leftover area.
	"""

	def __init__(self, width: int, height: int, placement:str="bottom_left") -> None:
		self.width = width
		self.height = height
		self._free: List[Rect] = [Rect(0, 0, width, height)]
		self.placed: List[PlacedPartitiotn] = []
		self.used_area: int = 0
		self.placement = placement

	@property
	def area(self) -> int:
		return self.width * self.height

	@property
	def utilization(self) -> float:
		if self.area == 0:
			return 0.0
		return self.used_area / self.area

	def can_fit(self, width: int, height: int) -> bool:
		return any(r.width >= width and r.height >= height for r in self._free)

	def try_insert(self, box_id: int, width: int, height: int, rotated: bool) -> bool:
		# use different placement heuristics based on self.placement
		if self.placement == "bottom_left":
			placement = self._find_bottom_left_placement(width, height)
		elif self.placement == "best_area_fit":
			placement = self._find_best_area_fit_placement(width, height)
		elif self.placement == "best_short_side_fit":
			placement = self._find_best_short_side_fit_placement(width, height)
		elif self.placement == "best_long_side_fit":
			placement = self._find_best_long_side_fit_placement(width, height)
		else:
			raise ValueError(f"Unknown placement heuristic: {self.placement}")

		if placement is None:
			return False
			
		placed_rect = Rect(placement.x, placement.y, width, height)
		self._split_free_rectangles(placed_rect)
		self._prune_free_list()

		self.placed.append(
			PlacedPartitiotn(
				partition_id=box_id,
				x=placed_rect.x,
				y=placed_rect.y,
				width=placed_rect.width,
				height=placed_rect.height,
				rotated=rotated,
			)
		)
		self.used_area += width * height
		return True
	
    # bottom_left placement heuristic: find the placement with the smallest y coordinate, and in case of tie, the smallest x coordinate, and in case of tie, the smallest leftover area
	def _find_bottom_left_placement(self, width: int, height: int) -> Optional[Rect]:
		best: Optional[Tuple[int, int, int, Rect]] = None
		w, h = width, height
		for free in self._free:
			if free.width < w or free.height < h:
				continue
			candidate = Rect(free.x, free.y, w, h)
			leftover_area = (free.width * free.height) - (w * h)
			key = (candidate.y, candidate.x, leftover_area)
			if best is None or key < (best[0], best[1], best[2]):
				best = (key[0], key[1], key[2], candidate)
		return best[3] if best else None
	
	# best_area_fit placement heuristic: find the placement with the smallest leftover area
	def _find_best_area_fit_placement(self, width: int, height: int) -> Optional[Rect]:
		best: Optional[Tuple[int, Rect]] = None
		w, h = width, height
		for free in self._free:
			if free.width < w or free.height < h:
				continue
			candidate = Rect(free.x, free.y, w, h)
			leftover_area = (free.width * free.height) - (w * h)
			if best is None or leftover_area < best[0]:
				best = (leftover_area, candidate)
		return best[1] if best else None

	# best_short_side_fit placement heuristic: find the placement with the smallest short side leftover, and in case of tie, the smallest long side leftover
	def _find_best_short_side_fit_placement(self, width: int, height: int) -> Optional[Rect]:
		best: Optional[Tuple[int, int, Rect]] = None
		w, h = width, height
		for free in self._free:
			if free.width < w or free.height < h:
				continue
			candidate = Rect(free.x, free.y, w, h)
			leftover_horiz = free.width - w
			leftover_vert = free.height - h
			short_side_fit = min(leftover_horiz, leftover_vert)
			long_side_fit = max(leftover_horiz, leftover_vert)
			key = (short_side_fit, long_side_fit)
			if best is None or key < (best[0], best[1]):
				best = (key[0], key[1], candidate)
		return best[2] if best else None
	
	def is_close_to_boundary(self, shape: ShapeInfo, partition_gap:int=5, boundary_tolerance: int = 3) -> bool:
		for placed_partition in self.placed:
			if shape.is_inside_roi((placed_partition.x + partition_gap + boundary_tolerance, placed_partition.y + partition_gap + boundary_tolerance, \
						   placed_partition.x + placed_partition.width - partition_gap - boundary_tolerance, placed_partition.y + placed_partition.height - partition_gap - boundary_tolerance)):
				return False
		return True


	# best_long_side_fit placement heuristic: find the placement with the smallest long side leftover, and in case of tie, the smallest short side leftover
	def _find_best_long_side_fit_placement(self, width: int, height: int) -> Optional[Rect]:
		best: Optional[Tuple[int, int, Rect]] = None
		w, h = width, height
		for free in self._free:
			if free.width < w or free.height < h:
				continue
			candidate = Rect(free.x, free.y, w, h)
			leftover_horiz = free.width - w
			leftover_vert = free.height - h
			short_side_fit = min(leftover_horiz, leftover_vert)
			long_side_fit = max(leftover_horiz, leftover_vert)
			key = (long_side_fit, short_side_fit)
			if best is None or key < (best[0], best[1]):
				best = (key[0], key[1], candidate)
		return best[2] if best else None

	def _split_free_rectangles(self, placed: Rect) -> None:
		new_free: List[Rect] = []
		for free in self._free:
			if not free.intersects(placed):
				new_free.append(free)
				continue

			# Split free rectangle into up to 4 rectangles around the placed one.
			if placed.x > free.x:
				new_free.append(Rect(free.x, free.y, placed.x - free.x, free.height))
			if placed.right < free.right:
				new_free.append(
					Rect(placed.right, free.y, free.right - placed.right, free.height)
				)
			if placed.y > free.y:
				new_free.append(Rect(free.x, free.y, free.width, placed.y - free.y))
			if placed.top < free.top:
				new_free.append(Rect(free.x, placed.top, free.width, free.top - placed.top))

		# Filter out degenerate rectangles
		self._free = [r for r in new_free if r.width > 0 and r.height > 0]

	def _prune_free_list(self) -> None:
		pruned: List[Rect] = []
		for rect in self._free:
			contained = False
			for other in self._free:
				if other is rect:
					continue
				if other.contains(rect):
					contained = True
					break
			if not contained:
				pruned.append(rect)
		self._free = pruned

	# from packed partition to bin frame
	def form_bin_frame(self, image: np.ndarray, partitions_by_id: Dict[int, Partition], merge_gap: int = 5) -> np.ndarray:
		bin_frame = np.zeros((self.width, self.height, 3), dtype=np.uint8)
		for placed_partition in self.placed:
			partition = partitions_by_id[placed_partition.partition_id]
			w_opsize, h_opsize = int(partition.size[0]), int(partition.size[1])
            # crop the partition from original image and place it on bin frame
			x_start = int(partition.center[0] - w_opsize//2)
			y_start = int(partition.center[1] - h_opsize//2)
			cropped_partition = image[y_start:y_start+h_opsize, x_start:x_start+w_opsize,:]
			if placed_partition.rotated:
				cropped_partition = np.transpose(cropped_partition, (1, 0, 2)) # rotate the cropped partition by 90 degree if the partition is rotated on BinPack
			x_offset, y_offset = int(placed_partition.x), int(placed_partition.y)
			x_end = x_offset + int(placed_partition.width)
			y_end = y_offset + int(placed_partition.height)
			bin_frame[int(y_offset+merge_gap):int(y_end-merge_gap), int(x_offset+merge_gap):int(x_end-merge_gap)] = cropped_partition
		return bin_frame
    
    # return results of bin frame to partition
    # prediction results is the list of points
	def decompose_prediction_results(self, shape_list: list[ShapeInfo], partition_by_id: Dict[int, Partition], boundary_tolerance:int=5) -> list[ShapeInfo]:
        # partition the results on bin frame based on the partition centers and opsize
		shape_list_on_original_coord = []
		for shape in shape_list:
			for placed_partition in self.placed:
				# check which partition the shape belongs to based on the IOU between the shape and the partition
				if shape.is_inside_roi((placed_partition.x + boundary_tolerance, placed_partition.y + boundary_tolerance, \
							placed_partition.x + placed_partition.width - boundary_tolerance, placed_partition.y + placed_partition.height - boundary_tolerance)):
					partition = partition_by_id[placed_partition.partition_id]
					center = (placed_partition.x + placed_partition.width//2, placed_partition.y + placed_partition.height//2)
					shape_on_partition_coord = deepcopy(shape)
					if placed_partition.rotated:
						# if the partition is rotated by 90 degree on BinPack, we need to rotate the shape back by -90 degree and then translate it back to original coordinate
						shape_on_partition_coord.translate(-center[0], -center[1])
						shape_on_partition_coord.transpose()
						shape_on_partition_coord.translate(placed_partition.height//2-boundary_tolerance, placed_partition.width//2-boundary_tolerance) # note the swap of x and y due to rotation
						shape_on_partition_coord.translate(partition.offset[0], partition.offset[1])
					else:
						shape_on_partition_coord.translate(-placed_partition.x-boundary_tolerance, -placed_partition.y-boundary_tolerance)
						shape_on_partition_coord.translate(partition.offset[0], partition.offset[1])
					shape_list_on_original_coord.append(shape_on_partition_coord)
					
		return shape_list_on_original_coord


Gene = Tuple[int, int]  # (partition_id, rotation_bit)
Chromosome = List[Gene]


def _decode_chromosome(
	chromosome: Chromosome,
	partitions_by_id: Dict[int, Partition],
	bin_width: int,
	bin_height: int,
	allow_rotation: bool,
	placement_strategy: str = "bottom_left",
	merge_gap: int = 5,
) -> Optional[List[MaxRectsBin]]:
	bins: List[MaxRectsBin] = []

	for partition_id, rot_bit in chromosome:
		partition = partitions_by_id[partition_id]
		rotated = bool(rot_bit) and allow_rotation
		w, h = (partition.size[1], partition.size[0]) if rotated else (partition.size[0], partition.size[1])
		w, h = int(w) + 2 * merge_gap, int(h) + 2 * merge_gap  # add merge gap to encourage merging nearby partitions

		if w > bin_width or h > bin_height:
			print(f"Partition {partition_id} with size ({w}, {h}) cannot fit into bin of size ({bin_width}, {bin_height}) even with rotation. Infeasible chromosome.")
			return None  # infeasible: box can never fit

		placed = False
		for b in bins:
			if b.try_insert(box_id=partition_id, width=w, height=h, rotated=rotated):
				placed = True
				break

		if not placed:
			new_bin = MaxRectsBin(bin_width, bin_height, placement=placement_strategy)
			if not new_bin.try_insert(box_id=partition_id, width=w, height=h, rotated=rotated):
				return None
			bins.append(new_bin)

	return bins


def _fitness(
	bins: List[MaxRectsBin],
	k: float = 2.0,
) -> float:
	"""Fitness to maximize.

	We want fewer partitions, but also reward dense packing.
	A practical scalarization:

		fitness = (sum(U_i^k) / N) / N  = sum(U_i^k) / N^2

	- Larger U_i improves fitness.
	- More bins (N) sharply reduces fitness.
	"""

	n = len(bins)
	if n == 0:
		return 0.0
	util_term = sum((b.utilization ** k) for b in bins) / n
	return util_term / n

#  Roulette Wheel Selection based on fitnesses. Higher fitness means higher chance to be selected.
def _roulette_wheel_select(
	population: Sequence[Chromosome],
	fitnesses: Sequence[float],
	rng: random.Random,
) -> Chromosome:
	total_fitness = sum(fitnesses)
	if total_fitness == 0:
		# If all fitnesses are zero, select randomly
		return list(rng.choice(population))
	selection_point = rng.uniform(0, total_fitness)
	current_sum = 0.0
	for chromo, fit in zip(population, fitnesses):
		current_sum += fit
		if current_sum >= selection_point:
			return list(chromo)
	# Fallback (should not happen if total_fitness > 0)
	return list(rng.choice(population))

# Tournament Selection: Randomly pick 'tournament_size' individuals and return the best among them.
def _tournament_select(
	population: Sequence[Chromosome],
	fitnesses: Sequence[float],
	tournament_size: int,
	rng: random.Random,
) -> Chromosome:
	best_idx = None
	best_fit = None
	for _ in range(tournament_size):
		idx = rng.randrange(len(population))
		fit = fitnesses[idx]
		if best_idx is None or fit > best_fit:  # type: ignore[operator]
			best_idx = idx
			best_fit = fit
	# return a shallow copy to avoid accidental in-place edits
	return list(population[best_idx])  # type: ignore[index]

# Stochastic Universal Sampling: Similar to roulette wheel but with multiple evenly spaced selection points, providing a more diverse selection.
def _stochastic_universal_sampling(
	population: Sequence[Chromosome],
	fitnesses: Sequence[float],num_selections: int,
	rng: random.Random,
) -> List[Chromosome]:
	total_fitness = sum(fitnesses)
	if total_fitness == 0:
		# If all fitnesses are zero, select randomly
		return [list(rng.choice(population)) for _ in range(num_selections)]
	start_point = rng.uniform(0, total_fitness / num_selections)
	pointers = [start_point + i * (total_fitness / num_selections) for i in range(num_selections)]

	selected: List[Chromosome] = []
	current_sum = 0.0
	idx = 0
	for chromo, fit in zip(population, fitnesses):
		current_sum += fit
		while idx < num_selections and current_sum >= pointers[idx]:
			selected.append(list(chromo))
			idx += 1
	return selected

def _ordered_crossover(parent1: Chromosome, parent2: Chromosome, rng: random.Random) -> Chromosome:
	"""Ordered crossover (OX) adapted to (id, rot_bit) genes."""

	n = len(parent1)
	if n <= 2:
		return list(parent1)

	a, b = sorted(rng.sample(range(n), 2))
	child: List[Optional[Gene]] = [None] * n

	# Build quick lookup for rotation bits
	rot1 = {partition_id: rot for partition_id, rot in parent1}
	rot2 = {partition_id: rot for partition_id, rot in parent2}

	slice_ids = []
	for i in range(a, b + 1):
		partition_id, rot = parent1[i]
		child[i] = (partition_id, rot)
		slice_ids.append(partition_id)

	slice_id_set = set(slice_ids)

	fill_positions = list(range(b + 1, n)) + list(range(0, a))
	p2_ids = [partition_id for partition_id, _ in parent2 if partition_id not in slice_id_set]

	p2_iter = iter(p2_ids)
	for pos in fill_positions:
		if child[pos] is not None:
			continue
		partition_id = next(p2_iter)
		child[pos] = (partition_id, rot2.get(partition_id, rot1.get(partition_id, 0)))

	# mypy: all positions filled
	return [g for g in child if g is not None]  # type: ignore[return-value]


def _swap_mutation(chromosome: Chromosome, rng: random.Random) -> None:
	if len(chromosome) < 2:
		return
	i, j = rng.sample(range(len(chromosome)), 2)
	chromosome[i], chromosome[j] = chromosome[j], chromosome[i]


def _inversion_mutation(chromosome: Chromosome, rng: random.Random) -> None:
	if len(chromosome) < 3:
		return
	i, j = sorted(rng.sample(range(len(chromosome)), 2))
	chromosome[i : j + 1] = reversed(chromosome[i : j + 1])


def _rotation_flip_mutation(chromosome: Chromosome, rng: random.Random) -> None:
	if not chromosome:
		return
	idx = rng.randrange(len(chromosome))
	partition_id, rot = chromosome[idx]
	chromosome[idx] = (partition_id, 1 - int(rot))


def partition_optimization_ga(
	partitions: List[Partition],
	bin_width: int,
	bin_height: int,
	population_size: int = 80,
	generations: int = 250,
	selection_strategy: str = "tournament",  # "roulette" or "tournament" or "sus"
	bin_insert_strategy: str = "bottom_left", # "bottom_left" or "best_area_fit" or "best_short_side_fit" or "best_long_side_fit"
	tournament_size: int = 3,
	sus_selection_num: int = 5,
	crossover_rate: float = 0.9,
	swap_mutation_rate: float = 0.20,
	inversion_mutation_rate: float = 0.10,
	rotation_flip_rate: float = 0.15,
	elitism: int = 2,
	utilization_power_k: float = 2.0,
	allow_rotation: bool = True,
	merge_gap: int = 5,
	seed: Optional[int] = 0,
	image: np.ndarray = None,
	log_folder: Optional[str] = None,
) -> Tuple[List[MaxRectsBin], float, Chromosome, List[float], List[float]]:
	"""Optimize partitions with a Genetic Algorithm.

	Args:
		partitions: List of partitions to be optimized.
		bin_width/bin_height: Fixed partition dimensions.
		population_size/generations: GA controls.
		allow_rotation: If True, each gene has a 0/1 flag to rotate a partition by 90°.
		seed: RNG seed for reproducibility. Use None for non-deterministic runs.

	Returns:
		(best_partitions, best_fitness, best_chromosome)
		- best_partitions: list of bins.
		- best_fitness: fitness value (higher is better).
		- best_chromosome: the GA chromosome that produced the solution.
		- best_fitness_history: list of best fitness values per generation (for analysis).
		- mean_fitness_history: list of mean fitness values per generation (for analysis).
	"""

	best_fitness_history: List[float] = []
	mean_fitness_history: List[float] = []

	rng = random.Random(seed)
	n = len(partitions)
	partitions_by_id = {b.id: b for b in partitions}

	if n == 0:
		return ([], 0.0, [])

	if population_size < 2:
		raise ValueError("population_size must be >= 2")
	if generations < 1:
		raise ValueError("generations must be >= 1")
	if elitism < 0 or elitism >= population_size:
		raise ValueError("elitism must be in [0, population_size-1]")
	
	for partition in partitions:
		w, h = partition.size
		if w+ merge_gap*2 > bin_width or h + merge_gap*2 > bin_height:
			raise ValueError(f"Partition {partition.id} with size ({w}, {h}) cannot fit into bin of size ({bin_width}, {bin_height}) even with rotation and merge gap. Infeasible input.")
	
	partition_ids = [b.id for b in partitions]

	def random_chromosome() -> Chromosome:
		perm = partition_ids[:]
		rng.shuffle(perm)
		if allow_rotation:
			return [(i, rng.randint(0, 1)) for i in perm]
		return [(i, 0) for i in perm]

	# Seeding: include a few sensible initial individuals (area-desc)
	by_area_desc = sorted(partitions, key=lambda b: b.area(), reverse=True)
	seeded: List[Chromosome] = []
	seeded.append([(b.id, 0) for b in by_area_desc])
	if allow_rotation:
		seeded.append([(b.id, 1) for b in by_area_desc])
	# A couple of shuffled variants of the sorted order
	for _ in range(2):
		perm = [b.id for b in by_area_desc]
		rng.shuffle(perm)
		seeded.append([(i, rng.randint(0, 1) if allow_rotation else 0) for i in perm])

	population: List[Chromosome] = []
	population.extend(seeded[: min(len(seeded), population_size)])
	while len(population) < population_size:
		population.append(random_chromosome())

	best_bins: List[MaxRectsBin] = None
	best_fit = float("-inf")
	best_chr: Chromosome = []

	for _gen in range(generations):
		decoded: List[MaxRectsBin] = [
			_decode_chromosome(c, partitions_by_id, bin_width, bin_height, allow_rotation, bin_insert_strategy, merge_gap)
			for c in population
		]
		fitnesses: List[float] = []
		for bins in decoded:
			if bins is None:
				fitnesses.append(float("-inf"))
			else:
				fitnesses.append(_fitness(bins, k=utilization_power_k))

		# Track best
		for c, bins, fit in zip(population, decoded, fitnesses):
			if fit > best_fit and bins is not None:
				best_fit = fit
				best_bins = bins
				best_chr = list(c)
				
		best_fitness_history.append(best_fit)
		mean_fit = sum(fitnesses) / len(fitnesses)
		mean_fitness_history.append(mean_fit)

		# Create next generation
		ranked = sorted(range(population_size), key=lambda i: fitnesses[i], reverse=True)
		next_population: List[Chromosome] = []
		for i in range(elitism):
			next_population.append(list(population[ranked[i]]))

		while len(next_population) < population_size:
			if selection_strategy == "roulette":
				p1 = _roulette_wheel_select(population, fitnesses, rng)
				p2 = _roulette_wheel_select(population, fitnesses, rng)
			elif selection_strategy == "tournament":
				p1 = _tournament_select(population, fitnesses, tournament_size, rng)
				p2 = _tournament_select(population, fitnesses, tournament_size, rng)
			elif selection_strategy == "sus":
				selected = _stochastic_universal_sampling(population, fitnesses, sus_selection_num, rng)
				# randomly pick 2 from the selected individuals
				p1, p2 = rng.sample(selected, 2)

			if rng.random() < crossover_rate:
				child = _ordered_crossover(p1, p2, rng)
			else:
				child = list(p1)

			if rng.random() < swap_mutation_rate:
				_swap_mutation(child, rng)
			if rng.random() < inversion_mutation_rate:
				_inversion_mutation(child, rng)
			if allow_rotation and rng.random() < rotation_flip_rate:
				_rotation_flip_mutation(child, rng)

			next_population.append(child)

		population = next_population

	if best_bins is None:
		# All individuals infeasible; return empty.
		return ([], float("-inf"), best_chr, best_fitness_history, mean_fitness_history)
	
	# debug chromosaome
	_decode_chromosome(best_chr, partitions_by_id, bin_width, bin_height, allow_rotation, bin_insert_strategy, merge_gap)
	
	if log_folder and image is not None:
		# save all merged image
		for i, b in enumerate(best_bins):
			bin_frame = b.form_bin_frame(image, partitions_by_id, merge_gap)
			cv2.imwrite(f"{log_folder}/best_bin_{i}.png", bin_frame)

	return (best_bins, best_fit, best_chr, best_fitness_history, mean_fitness_history)

# GA baseline
def partition_optimization_without_ga(
	partitions: List[Partition],
	bin_width: int,
	bin_height: int,
	allow_rotation: bool,
	placement_strategy: str = "bottom_left",
	utilization_power_k: float = 2.0,
	merge_gap: int = 5,
	image: np.ndarray = None,
	log_folder:str = None
) -> Optional[List[MaxRectsBin]]:
	"""Greedy partition optimization without GA, for baseline comparison."""
	for partition in partitions:
		w, h = partition.size
		if w+ merge_gap*2 > bin_width or h + merge_gap*2 > bin_height:
			raise ValueError(f"Partition {partition.id} with size ({w}, {h}) cannot fit into bin of size ({bin_width}, {bin_height}) even with rotation and merge gap. Infeasible input.")

	partitions_by_id = {b.id: b for b in partitions}
	# sort partitions by area descending
	desc_by_area = sorted(partitions, key=lambda b: b.area(), reverse=True)
	chromosome = [(b.id, 0) for b in desc_by_area]
	decoded = _decode_chromosome(chromosome, partitions_by_id, bin_width, bin_height, allow_rotation, placement_strategy)
	if decoded is None:
		return None, float("-inf"), chromosome
	fitness = _fitness(decoded, k=utilization_power_k) if decoded is not None else float("-inf")
	if log_folder and image is not None and decoded is not None:
		for i, b in enumerate(decoded):
			bin_frame = b.form_bin_frame(image, partitions_by_id)
			cv2.imwrite(f"{log_folder}/greedy_bin_{i}.png", bin_frame)
	return decoded, fitness, chromosome
