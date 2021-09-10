import dpgen.frontend.annotation as annotation

PREFIX = 'dpgen'
REPLACED_VARIABLE = f'{PREFIX}_replaced'
RANDOM_VARIABLE = f'{PREFIX}_eta'
LAMBDA_HOLE = f'{PREFIX}_lambda'
LAPLACE = f'{PREFIX}_lap'
OUTPUT = annotation.output.__name__
ASSERT = PREFIX + '_assert'
SAMPLE_INDEX = PREFIX + '_sample_index'
SAMPLE_ARRAY = PREFIX + '_sample_array'
HOLE = PREFIX + '_hole'
ALIGNMENT_ARRAY = PREFIX + '_theta'
ALIGNED_DISTANCE = PREFIX + '_aligned_distance'
SHADOW_DISTANCE = PREFIX + '_shadow_distance'
RANDOM_DISTANCE = PREFIX + '_random_distance'
SELECTOR = PREFIX + '_selector'
SELECT_ALIGNED = '0'
SELECT_SHADOW = '1'
V_EPSILON = PREFIX + '_v_epsilon'
INCREASING = 'INCREASING'
DECREASING = 'DECREASING'