import http from 'k6/http';
import { check } from 'k6';
import yaml from 'https://jslib.k6.io/jslib/jsyaml/4.1.0/index.js';

const configFile = open('./config.yaml');
const config = yaml.load(configFile);

// dynamic generation of the scenarios
const dynamicScenarios = {};
// counter for the different loads
let currentStartTime = 0;

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

  // when next load has to start = duration of current load + cooldown
  currentStartTime += config.load_test.duration_seconds + config.load_test.cooldown_seconds;
});

// Export dynamically operations generated
export const options = {
  scenarios: dynamicScenarios,
};

// loop of the load tests
export default function () {
  const url = `${config.endpoint.url}${config.request.path}`;

  // Creating dummy payload (consider max number as item id)
  const payload = {
    sequence_length: config.request.sequence_length,
    features: Array.from({ length: config.request.sequence_length }, () => Math.floor(Math.random() * 100)),
  };

  const params = {
    headers: Object.assign(
      { 'Content-Type': 'application/json' },
      config.headers || {}
    ),
    timeout: `${config.endpoint.client.read_timeout * 1000}`,
  };

  const res = http.post(url, JSON.stringify(payload), params);

  check(res, {
    'status is 200': (r) => r.status === 200,
  });
}