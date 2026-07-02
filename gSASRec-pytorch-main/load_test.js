// to run:
// k6 run load_test.js --summary-export=results.json

import { check } from 'k6';
import http from 'k6/http';

const config = JSON.parse(open('./config.json'));

// dynamic generation of the scenarios
const dynamicScenarios = {};
// dynamic generation of the thresholds for the warmup
const dynamicThresholds = {};
const warmupDurationSeconds = 15;
// counter for the different loads
let currentStartTime = 0;

// warmup scenario with lower rate
dynamicScenarios['warmup'] = {
  executor: 'constant-arrival-rate',
  rate: config.load_test.target_rps[0],
  timeUnit: '1s',
  duration: `${warmupDurationSeconds}s`,
  startTime: '0s',
  preAllocatedVUs: 10,
  maxVUs: config.load_test.num_clients,
};

currentStartTime = warmupDurationSeconds + config.load_test.cooldown_seconds;

config.load_test.target_rps.forEach((rps) => {
  const stepName = `step_${rps}_rps`;

  dynamicScenarios[stepName] = {
    executor: 'constant-arrival-rate',
    rate: rps,
    timeUnit: '1s',
    duration: `${config.load_test.duration_seconds}s`,
    startTime: `${currentStartTime}s`,

    // num_clients as max number of threads
    preAllocatedVUs: Math.min(rps, config.load_test.num_clients),
    maxVUs: config.load_test.num_clients,
  };
  // trick to compute isolated percentiles for this specific step
  dynamicThresholds[`http_req_duration{scenario:${stepName}}`] = ['p(95)>=0'];

  // when next load has to start = duration of current load + cooldown
  currentStartTime += config.load_test.duration_seconds + config.load_test.cooldown_seconds;
});

// Export dynamically operations generated
export const options = {
  scenarios: dynamicScenarios,
  thresholds: dynamicThresholds, // filters to see clean percentiles
};

// loop of the load tests
export default function () {
  const url = `${config.endpoint.url}${config.request.path}`;

  const batchSize = 16;
  const sequences = [];

  for (let i = 0; i < batchSize; i++) {
    const seq = Array.from(
      { length: config.request.sequence_length },
      () => Math.floor(Math.random() * 3415) + 1
    );
    sequences.push(seq);
  }

  const payload = {
    batch_sequences: sequences
  }

  const params = {
    headers: Object.assign({ 'Content-Type': 'application/json' }, config.headers || {}),
    timeout: `${config.endpoint.client.read_timeout * 1000}`,
  };

  const res = http.post(url, JSON.stringify(payload), params);

  check(res, {
    'status is 200': (r) => r.status === 200,
  });
}