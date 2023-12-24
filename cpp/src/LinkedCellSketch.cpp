#include "LinkedCellSketch.h"

LinkedCellSketch::LinkedCellSketch(int width, int depth, int split_counter_width) : Dictionary(),
                                                                                    split_counter_width(split_counter_width),
                                                                                    counters(depth, std::vector<Row>())
{
    size = 1;
    for (int i = 0; i < depth; i++)
    {
        appendRow(i, width);
    }
}

LinkedCellSketch::~LinkedCellSketch()
{
}

void LinkedCellSketch::update(uint32_t key, int amount)
{
    for (int depth = 0; depth < counters.size(); depth++)
    {
        auto &current_row = counters[depth];
        int inner_row_index = 0;
        while (true)
        {
            auto &inner_row = current_row[inner_row_index];
            int cell_index = getIndexInRow(depth, key, inner_row_index);
            auto &cell = inner_row[cell_index];
            if (!cell.index)
            {
                // overflow?
                cell.value += amount;
                break;
            }
            inner_row_index = cell.index;
        }
    }
}

int LinkedCellSketch::query(uint32_t key)
{
    int estimate = INT_MAX;
    for (int depth = 0; depth < counters.size(); depth++)
    {
        int current_estimate = 0;
        int inner_row_index = 0;
        while (true)
        {
            const auto &row = counters[depth][inner_row_index];
            int cell_index = getIndexInRow(depth, key, inner_row_index);
            const auto &cell = row[cell_index];
            current_estimate += cell.value;
            if (!cell.index)
            {
                break;
            }
            inner_row_index = cell.index;
        }
        estimate = std::min(estimate, current_estimate);
    }
    return estimate;
}

void LinkedCellSketch::getLeaves(int depth, std::vector<int> leaves_indice) const
{
    leaves_indice.clear();
    for (int row_index = 1; row_index < counters[depth].size(); row_index++)
    {
        const auto &row = counters[depth][row_index];
        bool is_leaf = true;
        for (const auto &cell : row)
        {
            if (cell.index)
            {
                is_leaf = false;
                break;
            }
        }
        if (is_leaf)
        {
            leaves_indice.push_back(row_index);
        }
    }
}

template <class T, class Compare = std::less<>>
void sortedInsertCapped(std::vector<T> &vec, const T &val, std::size_t size_limit, Compare comp = Compare{})
{
    vec.insert(std::upper_bound(vec.begin(), vec.end(), val, comp), val);
    if (vec.size() > size_limit)
    {
        vec.erase(vec.begin() + size_limit, vec.end());
    }
}

void LinkedCellSketch::expand()
{
    size++;
    int depth = counters.size();
    int width = counters[0][0].size();
    int expanded_counters = width * depth / split_counter_width;
    struct CellLoad
    {
        uint32_t cell_load;
        uint32_t depth_index;
        uint32_t row_index;
        uint32_t cell_index;
    };
    std::vector<CellLoad> cells_load;
    for (uint32_t depth = 0; depth < counters.size(); depth++)
    {
        for (uint32_t row_index = 0; row_index < counters[depth].size(); row_index++)
        {
            const auto &row = counters[depth][row_index];
            for (uint32_t cell_index = 0; cell_index < row.size(); cell_index++)
            {
                const auto &cell = row[cell_index];
                if (!cell.index)
                {
                    CellLoad cell_load{cell.value, depth, row_index, cell_index};
                    sortedInsertCapped(
                        cells_load,
                        cell_load,
                        expanded_counters,
                        [](const auto &cl0, const auto &cl1)
                        {
                            if (cl0.row_index == cl1.row_index){
                                return cl0.cell_load > cl1.cell_load;
                            }
                            return cl0.row_index < cl1.row_index;
                        });
                }
            }
        }
    }
    for (const auto &cell_load : cells_load)
    {
        auto &cell = counters[cell_load.depth_index][cell_load.row_index][cell_load.cell_index];
        cell.index = appendRow(cell_load.depth_index, split_counter_width);
    }
    //printRows();
}

void LinkedCellSketch::shrink()
{
    size--;
    int depth = counters.size();
    int width = counters[0].size();
    int expanded_counters = width * depth / split_counter_width;
    for (int depth = 0; depth < counters.size(); depth++)
    {
        std::vector<std::pair<int, int>> rows_load;
        std::vector<int> leaves_indice;
        getLeaves(depth, leaves_indice);
        for (const auto &leaf_indice : leaves_indice)
        {
            const auto &row = counters[depth][leaf_indice];
            int row_load = 0;
            for (const auto &cell : row)
            {
                row_load += cell.value;
            }
            rows_load.push_back(std::make_pair(leaf_indice, row_load));
        }
        std::sort(
            rows_load.begin(),
            rows_load.end(),
            [](const auto &p0, const auto &p1)
            {
                return p0.second < p1.second;
            });
        throw std::runtime_error("LinkedCellSketch::shrink - implement remove leaves.");
    }
}

int LinkedCellSketch::appendRow(int depth, int cell_count)
{
    int row_index = counters[depth].size();
    counters[depth].push_back(std::vector<Cell>(cell_count));
    return row_index;
}

int LinkedCellSketch::getSize() const
{
    return size;
}

int LinkedCellSketch::getMemoryUsage() const
{
    int result = 0;
    for (int depth = 0; depth < counters.size(); depth++)
    {
        const auto &rows = counters[depth];
        result += 9 * rows[0].size() + 9 * split_counter_width * (rows.size() - 1);
    }
    // result is in bits, we want bytes
    return result / 8;
}

void LinkedCellSketch::printRows() const
{
    for (int depth = 0; depth < counters.size(); depth++)
    {
        std::cout << "depth:" << depth << ", ";
        const auto &rows = counters[depth];
        for (int row_index = 0; row_index < rows.size(); row_index++)
        {
            std::cout << "row:" << row_index << std::endl;
            const auto &row = rows[row_index];
            for (int i = 0; i < row.size(); i++)
            {
                const auto cell = &row[i];
                std::cout << "counter:" << i << ", index:" << cell->index << ", value:" << (int)cell->value << std::endl;
            }
        }
    }
}

int LinkedCellSketch::getIndexInRow(int depth, uint32_t key, int row_index) const
{
    return murmurhash((int *)&key, depth + row_index) % counters[depth][row_index].size();
}
