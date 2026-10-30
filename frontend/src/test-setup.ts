import "@testing-library/jest-dom/vitest";

let urlCounter = 0;
URL.createObjectURL = () => `blob:test-${urlCounter++}`;
URL.revokeObjectURL = () => {};
