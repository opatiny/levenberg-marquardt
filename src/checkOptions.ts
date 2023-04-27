import { isAnyArray } from 'is-any-array';

export interface Data<SrcType, DstType> {
  source: SrcType[];
  destination: DstType[];
}

export type FitFunction<SrcType, DstType> = (value: SrcType) => DstType;

export interface LevenbergMarquardtOptions {
  timeout: number;
  minValues: number[];
  maxValues: number[];
  initialValues: number[];
  weights?: number | number[];
  damping?: number;
  dampingStepUp?: number;
  dampingStepDown?: number;
  maxIterations?: number;
  errorTolerance?: number;
  centralDifference?: boolean;
  gradientDifference?: number | number[];
  improvementThreshold?: number;
}

export default function checkOptions<SrcType, DstType>(
  data: Data<SrcType, DstType>,
  parameterizedFunction,
  options: LevenbergMarquardtOptions,
) {
  let {
    timeout,
    minValues,
    maxValues,
    initialValues,
    weights = 1,
    damping = 1e-2,
    dampingStepUp = 11,
    dampingStepDown = 9,
    maxIterations = 100,
    errorTolerance = 1e-7,
    centralDifference = false,
    gradientDifference = 10e-2,
    improvementThreshold = 1e-3,
  } = options;

  if (damping <= 0) {
    throw new Error('The damping option must be a positive number');
  } else if (
    !isAnyArray(data.source) ||
    data.source.length < 2 ||
    !isAnyArray(data.destination) ||
    data.destination.length < 2
  ) {
    throw new Error(
      'The data parameter elements must be arrays with more than 2 elements',
    );
  } else if (data.source.length !== data.destination.length) {
    throw new Error('The data source and destination must have the same size');
  }

  if (!(initialValues && initialValues.length > 0)) {
    throw new Error(
      'The initialValues option is mandatory and must be an array',
    );
  }
  let parameters = initialValues;

  let nbPoints = data.destination.length;
  let parLen = parameters.length;
  maxValues = maxValues || new Array(parLen).fill(Number.MAX_SAFE_INTEGER);
  minValues = minValues || new Array(parLen).fill(Number.MIN_SAFE_INTEGER);

  if (maxValues.length !== minValues.length) {
    throw new Error('minValues and maxValues must be the same size');
  }

  let finalGradientDifference: number[] = [];
  if (typeof gradientDifference === 'number') {
    finalGradientDifference = new Array(parameters.length).fill(
      gradientDifference,
    );
  } else if (isAnyArray(gradientDifference)) {
    if (gradientDifference.length !== parLen) {
      finalGradientDifference = new Array(parLen).fill(gradientDifference[0]);
    }
  } else {
    throw new Error(
      'gradientDifference should be a number or array with length equal to the number of parameters',
    );
  }

  let filler;
  if (typeof weights === 'number') {
    let value = 1 / weights ** 2;
    filler = () => value;
  } else if (isAnyArray(weights) && weights.length === data.source.length) {
    if (weights.length < data.source.length) {
      let value = 1 / weights[0] ** 2;
      filler = () => value;
    } else {
      filler = (i) => 1 / weights[i] ** 2;
    }
  } else {
    throw new Error(
      'weights should be a number or an array with length equal to the number of data points',
    );
  }

  let weightSquare = new Array(data.source.length);
  for (let i = 0; i < nbPoints; i++) {
    weightSquare[i] = filler(i);
  }

  let checkTimeout: () => boolean;
  if (timeout !== undefined) {
    if (typeof timeout !== 'number') {
      throw new Error('timeout should be a number');
    }
    let endTime = Date.now() + timeout * 1000;
    checkTimeout = () => Date.now() > endTime;
  } else {
    checkTimeout = () => false;
  }

  return {
    checkTimeout,
    minValues,
    maxValues,
    parameters,
    weightSquare,
    damping,
    dampingStepUp,
    dampingStepDown,
    maxIterations,
    errorTolerance,
    centralDifference,
    gradientDifference: finalGradientDifference,
    improvementThreshold,
  };
}
