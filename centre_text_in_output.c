/*
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>

int main(void) {
    int terminal_width = 80;  // default width if retrieval fails
    struct winsize ws;

    // Retrieve the terminal window size.
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != -1) {
        terminal_width = ws.ws_col;
    }

    const char *message = "Hello, World! This text is centered.";
    int message_length = strlen(message);

    // Calculate the number of spaces needed on the left.
    int padding = (terminal_width - message_length) / 2;
    if (padding < 0) {
        padding = 0;
    }

    // Print the left padding.
    for (int i = 0; i < padding; i++) {
        putchar(' ');
    }
    // Print the centered message.
    printf("%s\n", message);

    return 0;
}
*//*
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>

int main(void) {
    int terminal_width = 80;  // default width if retrieval fails
    struct winsize ws;
    
    // Try to get the terminal size (works on macOS)
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != -1) {
        terminal_width = ws.ws_col;
    }
    
    const char *message = "Hello, World! This text is centered.";
    int message_length = strlen(message);
    
    // Calculate the number of spaces needed on the left.
    int padding = (terminal_width - message_length) / 2;
    if (padding < 0) {
        padding = 0;
    }
    
    // Use printf's field width specifier:
    // "%*s" prints a string in a field of a given width.
    // Here, we print an empty string with the desired padding width.
    printf("%*s%s\n", padding, "", message);
    
    return 0;
}
*//*
Below is a comparison of the two approaches—the explicit loop and the printf field-width specifier method—focusing on their runtime efficiency, time complexity, space complexity, and some back‐of‐the‐envelope calculations.

---

### **Approach 1: Explicit Loop**

**Code Sketch:**

```c
for (int i = 0; i < padding; i++) {
putchar(' ');
}
printf("%s\n", message);
```

**Time Complexity:**

- The loop runs exactly `padding` times. If we call the number of spaces `P` and the message length `M`, then:
- The loop takes O(P) time.
- Printing the message takes O(M) time.
- **Overall:** O(P + M)

**Space Complexity:**

- The loop uses only a few integer variables (the counter, etc.), which is O(1) extra space.
- No additional dynamic memory is allocated.

**Runtime Considerations:**

- Every iteration of the loop calls `putchar(' ')`. In C, `putchar` may involve function-call overhead (even though it’s usually inlined or buffered) and may incur a per-character overhead in the worst case.
- For a terminal width of, say, 80 characters and a message length of 40, you’d loop 20 times (if padding = (80–40)/2 = 20).  
- If each iteration (including the function call overhead) takes, for example, 5 CPU cycles, then the loop costs roughly 20 × 5 = 100 cycles (plus the cost of printing the message).

---

### **Approach 2: printf Field-Width Specifier**

**Code Sketch:**

```c
printf("%*s%s\n", padding, "", message);
```

**Time Complexity:**

- The `%*s` specifier tells `printf` to print an empty string in a field of width `padding`. Internally, the standard library must generate `padding` spaces.
- Although the internal implementation isn’t specified by the standard, most C libraries use highly optimized routines (often in assembly or using efficient memory-setting functions) to handle such formatting.
- **Overall:** O(P + M) as well, since it still must output `P` spaces and `M` characters of the message.

**Space Complexity:**

- Like the loop approach, this method uses only constant extra space (a few registers and variables inside `printf`).
- No additional heap allocation is required.

**Runtime Considerations:**

- Instead of a loop in user code, this method delegates the work to the library’s formatted output routines.  
- Such routines are typically optimized (e.g., they might call a routine similar to `memset` to fill a buffer with spaces before outputting it).
- This may reduce per-iteration overhead compared to repeatedly calling `putchar()`.
- For the same padding of 20 spaces and message length of 40, the internal implementation might perform fewer function-call checks per character, leading to a lower constant factor.

---

### **Comparison Summary**

1. **Time Complexity:**
- **Both methods** have O(P + M) time complexity.
- In practice, the *printf* approach likely has a lower constant overhead because the library function can be highly optimized and may perform the operation in fewer instructions or even a single call to a memory-fill routine.

2. **Space Complexity:**
- Both methods use constant extra space, O(1).

3. **Practical Efficiency:**
- **Explicit Loop:**  
- More straightforward to understand and debug.
- Involves a C-level loop and individual calls to `putchar()`, which might be less efficient if the loop is large.
- **Printf Field-Width:**  
- More concise and leverages the optimized internal implementation of the C standard library.
- Likely faster for large padding widths because it avoids the overhead of a loop in user code.
- **Example Calculation:**  
- If padding P = 20 and each loop iteration in the explicit loop costs ~5 cycles, then the loop costs 100 cycles.  
- The printf version might, for example, fill a buffer of 20 characters in 20 cycles (or even less if optimized with a vectorized memset-like function) plus the overhead of one function call.  
- In scenarios where P is large (say, hundreds of characters), the difference becomes more significant.

---

### **Conclusion**

- **Both approaches scale linearly** with the number of padding spaces and the length of the message.
- The **printf field-width specifier** is likely to be more efficient in practice due to lower overhead in highly optimized C library implementations.
- The explicit loop is simpler to understand at a basic level but may incur more overhead due to the loop and per-character function calls.
- **For most terminal output scenarios,** the difference in runtime is negligible, but if you imagine an “alien super programmer” optimizing for every cycle, the printf approach with its optimized internal routines is the better choice.

This comprehensive comparison shows that while both methods are O(P + M) in time and O(1) in space, the printf field-width specifier can be more efficient in practice due to lower constant factors.
s