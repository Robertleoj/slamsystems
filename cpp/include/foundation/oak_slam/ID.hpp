#include <memory>

namespace foundation {
namespace oak_slam {

template <typename Tag>
struct ID {
    using IdType = uint64_t;
    IdType value;
    static std::atomic<IdType> counter;  // Declare static member

    explicit ID(
        IdType v
    )
        : value(v) {}  // Manual ID creation

    static ID next() { return ID(counter++); }  // Generate unique ID

    bool operator==(
        const ID& other
    ) const {
        return value == other.value;
    }
    bool operator!=(
        const ID& other
    ) const {
        return value != other.value;
    }
    bool operator<(
        const ID& other
    ) const {
        return value < other.value;
    }
};

// Define static counter outside the struct
template <typename Tag>
std::atomic<typename ID<Tag>::IdType> ID<Tag>::counter{1};

}  // namespace oak_slam
}  // namespace foundation

namespace std {
template <typename Tag>
struct hash<foundation::oak_slam::ID<Tag>> {
    size_t operator()(
        const foundation::oak_slam::ID<Tag>& id
    ) const {
        return hash<uint64_t>()(id.value);
    }
};
}  // namespace std
