import "@testing-library/jest-dom";

// Mock básico de fetch para tests; cada test lo redefinirá según necesidad.
beforeEach(() => {
  // @ts-ignore
  global.fetch = vi.fn();
});

afterEach(() => {
  vi.restoreAllMocks();
});
