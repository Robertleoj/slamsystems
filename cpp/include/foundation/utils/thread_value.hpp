#pragma once
#include <atomic>
#include <mutex>
#include <optional>

namespace foundation {

/**
 * equivalent to a queue with a max size of 1.
 */
template <typename T>
class ThreadValue {
   public:
    void update(
        T& val
    ) {
        {
            std::scoped_lock l(mtx);
            curr_value = val;
        }
        updated = true;
    }

    std::optional<T> get_update() {
        if (!updated) {
            return std::nullopt;
        }

        std::scoped_lock l(mtx);

        updated = false;
        return curr_value.value();
    }

   private:
    std::mutex mtx;
    std::atomic<bool> updated;

    std::optional<T> curr_value;
};
}  // namespace foundation