// todo: test all checkOptions errors

import checkOptions, { Data } from '../checkOptions';

const basicData: Data<number, number> = {
  source: [1, 2, 3],
  destination: [4, 5, 6],
};
const line = (value: number) => {
  return value;
};
test('timeout is not a number error', () => {
  expect(checkOptions(basicData, line, { timeout: -5 }));
});
