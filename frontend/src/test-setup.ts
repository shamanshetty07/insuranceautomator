import "@testing-library/jest-dom/vitest";

// jsdom lacks object URLs; tests only need unique, revocable strings.
let urlCounter = 0;
URL.createObjectURL = () => `blob:test-${urlCounter++}`;
URL.revokeObjectURL = () => {};
