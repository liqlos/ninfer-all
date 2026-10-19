#pragma once

namespace ninfer::ops::gguf {

struct Q6MatrixPartition {
    int parts;
    int row_tile;
};

Q6MatrixPartition q6_matrix_partition(int rows, int columns, int maximum_rows);

}
