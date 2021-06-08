'''In this file you need to implement remote procedure call (RPC) server

* There are different RPC libraries for python, such as xmlrpclib, json-rpc. You are free to choose.
* The following functions have to be implemented and exported:
 * get_angle
 * set_angle
 * get_posture
 * execute_keyframes
 * get_transform
 * set_transform
* You can test RPC server with ipython before implementing agent_client.py
'''

# add PYTHONPATH
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'kinematics'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))

from kinematics.inverse_kinematics import InverseKinematicsAgent
import threading
import pickle
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


class ServerAgent(InverseKinematicsAgent):
    '''ServerAgent provides RPC service
    '''

    # YOUR CODE HERE

    def __init__(self):
        super(ServerAgent, self).__init__()
        robot_pose_data = open('robot_pose.pkl', 'rb')
        self.posture_classifier = pickle.load(robot_pose_data)  # LOAD YOUR CLASSIFIER
        self.posture = 'unknown'
        robot_pose_data.close()
        # create server
        self.server = SimpleXMLRPCServer(('localhost', 8000), requestHandler=RequestHandler, allow_none=True)
        self.server.register_introspection_functions()
        self.server.register_multicall_functions()
        self.server.register_instance(self)
        print('created a server')
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        print('Server started')

    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        return super(ServerAgent, self).think(perception)

    def recognize_posture(self, perception):
        posture = 'unknown'
        # YOUR CODE HERE
        joint_names = ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'RHipYawPitch', 'RHipRoll', 'RHipPitch',
                       'RKneePitch']
        values = []
        for joint_name in joint_names:
            values.append(perception.joint[joint_name])
        values += perception.imu

        postures = ['HeadBack', 'Left', 'Right', 'Crouch', 'Knee', 'Stand', 'Sit', 'StandInit', 'Frog', 'Back', 'Belly']
        posture_id = self.posture_classifier.predict([values])[0]
        posture = postures[posture_id]

        return posture

    def get_angle(self, joint_name):
        '''get sensor value of given joint'''
        return self.perception.joint.get(joint_name)

    def set_angle(self, joint_name, angle):
        '''set target angle of joint for PID controller
        '''
        # YOUR CODE HERE
        if joint_name in self.perception.joint:
            self.target_joints[joint_name] = angle
            print('Setting angle ', joint_name, 'to ', str(angle))
        else:
            print('Error: Could not set angle')

    def get_posture(self):
        '''return current posture of robot'''
        # YOUR CODE HERE
        print("Robot has posture = ", self.posture)
        return str(self.posture)

    def execute_keyframes(self, keyframes):
        '''excute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        '''
        # YOUR CODE HERE
        return self.keyframes(keyframes)

    def get_transform(self, name):
        '''get transform with given name
        '''
        # YOUR CODE HERE
        return self.transforms[name]

    def set_transform(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        self.transforms[effector_name] = transform


if __name__ == '__main__':
    agent = ServerAgent()
    agent.run()
