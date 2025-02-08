
/*
* comprehensive_demo_part1.c
*
* A comprehensive demonstration of important C programming concepts:
* - Basic data types (int, float, double, char, long, unsigned)
* - Structures and typedefs
* - Pointers and pointer arithmetic
* - Dynamic memory allocation
* - Search algorithms (linear and binary search)
* - Sorting algorithms (bubble sort and qsort)
* - Recursion (factorial)
* - Function pointers
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* --- Structure Definitions --- */

typedef struct {
	char name[50];
	int age;
} Person;

/* --- Function Prototypes --- */

/* Demonstration functions */
void demoDataTypes(void);
void demoStructs(void);
void demoPointers(void);
void demoDynamicMemory(void);
void demoSorting(void);
void demoSearching(void);
void demoRecursion(void);
void demoFunctionPointers(void);

/* Utility functions for structs and arrays */
int comparePersons(const void *a, const void *b);

/* Sorting and Searching Algorithms */
void bubbleSort(int arr[], int n);
int cmp_int(const void *a, const void *b);
int linearSearch(int arr[], int n, int target);
int binarySearch(int arr[], int n, int target);

/* Recursion */
int factorial(int n);

/* --- Main Function --- */

int main(void) {
	printf("=== Comprehensive C Program Demonstration ===\n\n");
	
	demoDataTypes();
	demoStructs();
	demoPointers();
	demoDynamicMemory();
	demoSorting();
	demoSearching();
	demoRecursion();
	demoFunctionPointers();
	
	return 0;
}

/* --- Function Implementations --- */

/* 1. Demonstrate basic data types */
void demoDataTypes(void) {
	printf("Demo: Basic Data Types\n");
	
	int i = 42;
	float f = 3.14f;
	double d = 2.71828;
	char c = 'A';
	long l = 1234567890L;
	unsigned int ui = 4000000000U;
	
	printf("int: %d\n", i);
	printf("float: %.2f\n", f);
	printf("double: %.5f\n", d);
	printf("char: %c (ASCII %d)\n", c, c);
	printf("long: %ld\n", l);
	printf("unsigned int: %u\n\n", ui);
}

/* 2. Demonstrate structs and sorting an array of structs */
void demoStructs(void) {
	printf("Demo: Structures (structs)\n");
	
	Person people[3] = {
		{"Alice", 30},
		{"Bob", 25},
		{"Charlie", 35}
	};
	
	printf("Original list of people:\n");
	for (int i = 0; i < 3; i++) {
		printf("  Name: %s, Age: %d\n", people[i].name, people[i].age);
	}
	
	// Sort by age using qsort
	qsort(people, 3, sizeof(Person), comparePersons);
	
	printf("\nAfter sorting by age (using qsort):\n");
	for (int i = 0; i < 3; i++) {
		printf("  Name: %s, Age: %d\n", people[i].name, people[i].age);
	}
	printf("\n");
}

/* Compare function for qsort for Person, sorting by age */
int comparePersons(const void *a, const void *b) {
	const Person *p1 = (const Person *)a;
	const Person *p2 = (const Person *)b;
	return (p1->age - p2->age);
}

/* 3. Demonstrate pointers and pointer arithmetic */
void demoPointers(void) {
	printf("Demo: Pointers and Pointer Arithmetic\n");
	
	int array[5] = {10, 20, 30, 40, 50};
	int *p = array;  // pointer to the first element of array
	
	printf("Array elements using pointer arithmetic:\n");
	for (int i = 0; i < 5; i++) {
		printf("  *(p + %d) = %d\n", i, *(p + i));
	}
	printf("\n");
}

/* 4. Demonstrate dynamic memory allocation */
void demoDynamicMemory(void) {
	printf("Demo: Dynamic Memory Allocation\n");
	
	int n = 5;
	int *dynArray = (int *)malloc(n * sizeof(int));
	if (dynArray == NULL) {
		printf("Memory allocation failed!\n");
		return;
	}
	
	for (int i = 0; i < n; i++) {
		dynArray[i] = (i + 1) * 10;
	}
	
	printf("Dynamically allocated array: ");
	for (int i = 0; i < n; i++) {
		printf("%d ", dynArray[i]);
	}
	printf("\n");
	
	free(dynArray);
	printf("\n");
}

/* 5. Demonstrate sorting algorithms */
void demoSorting(void) {
	printf("Demo: Sorting Algorithms\n");
	
	int arr1[] = {42, 23, 4, 16, 8, 15};
	int n1 = sizeof(arr1) / sizeof(arr1[0]);
	
	// Use bubble sort on arr1
	bubbleSort(arr1, n1);
	
	printf("Array sorted using bubble sort: ");
	for (int i = 0; i < n1; i++) {
		printf("%d ", arr1[i]);
	}
	printf("\n");
	
	// Use qsort on another array
	int arr2[] = {42, 23, 4, 16, 8, 15};
	int n2 = sizeof(arr2) / sizeof(arr2[0]);
	qsort(arr2, n2, sizeof(int), cmp_int);
	
	printf("Array sorted using qsort: ");
	for (int i = 0; i < n2; i++) {
		printf("%d ", arr2[i]);
	}
	printf("\n\n");
}

/* Bubble sort implementation */
void bubbleSort(int arr[], int n) {
	bool swapped;
	for (int i = 0; i < n - 1; i++) {
		swapped = false;
		for (int j = 0; j < n - i - 1; j++) {
			if (arr[j] > arr[j + 1]) {
				int temp = arr[j];
				arr[j] = arr[j + 1];
				arr[j + 1] = temp;
				swapped = true;
			}
		}
		if (!swapped)
			break;
	}
}

/* Comparison function for qsort on ints */
int cmp_int(const void *a, const void *b) {
	int arg1 = *(const int *)a;
	int arg2 = *(const int *)b;
	if (arg1 < arg2)
		return -1;
	if (arg1 > arg2)
		return 1;
	return 0;
}

/* 6. Demonstrate searching algorithms */
void demoSearching(void) {
	printf("Demo: Searching Algorithms\n");
	
	int arr[] = {4, 16, 8, 15, 23, 42};
	int n = sizeof(arr) / sizeof(arr[0]);
	int target = 15;
	
	// Linear search (unsorted array)
	int indexLinear = linearSearch(arr, n, target);
	if (indexLinear != -1) {
		printf("  Linear Search: Found %d at index %d\n", target, indexLinear);
	} else {
		printf("  Linear Search: %d not found\n", target);
	}
	
	// For binary search, sort a copy of the array first.
	int arrSorted[sizeof(arr) / sizeof(arr[0])];
	memcpy(arrSorted, arr, sizeof(arr));
	bubbleSort(arrSorted, n);
	
	int indexBinary = binarySearch(arrSorted, n, target);
	if (indexBinary != -1) {
		printf("  Binary Search: Found %d at index %d in the sorted array\n", target, indexBinary);
	} else {
		printf("  Binary Search: %d not found in the sorted array\n", target);
	}
	printf("\n");
}

/* Linear search implementation */
int linearSearch(int arr[], int n, int target) {
	for (int i = 0; i < n; i++) {
		if (arr[i] == target)
			return i;
	}
	return -1;
}

/* Binary search implementation (iterative) on a sorted array */
int binarySearch(int arr[], int n, int target) {
	int low = 0, high = n - 1;
	while (low <= high) {
		int mid = low + (high - low) / 2;
		if (arr[mid] == target)
			return mid;
		else if (arr[mid] < target)
			low = mid + 1;
		else
			high = mid - 1;
	}
	return -1;
}

/* 7. Demonstrate recursion */
void demoRecursion(void) {
	printf("Demo: Recursion (Factorial)\n");
	
	int num = 5;
	int fact = factorial(num);
	printf("  Factorial of %d is %d\n\n", num, fact);
}

/* Recursive factorial function */
int factorial(int n) {
	if (n <= 1)
		return 1;
	return n * factorial(n - 1);
}

/* 8. Demonstrate function pointers */
void demoFunctionPointers(void) {
	printf("Demo: Function Pointers\n");
	
	/* Define a function pointer for a function that takes an int and returns an int.
	We use it to point to our factorial function. */
	int (*funcPtr)(int) = factorial;
	
	int num = 6;
	int result = funcPtr(num);
	printf("  Using function pointer: Factorial of %d is %d\n\n", num, result);
}

