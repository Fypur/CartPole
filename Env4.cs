using AI;
using Fiourp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CartPole
{
    public class Env4
    {
        public Cart Cart;
        public Pole Pole;
        private ActorCritic agent;

        private int timeStep = 0;
        private int maxDistanceCar = 200;

        int batchSize = 10;
        List<float[]> states = new();
        List<float> rewards = new();
        List<int> actions = new();
        List<int> episodeLengths = new();

        public Env4()
        {
            this.agent = new ActorCritic(new int[] { 4, 32, 32, 3 });
            Cart = new Cart();
            Pole = new Pole(Cart);

            Cart.Reset();
            Pole.Reset();
        }

        public void Update()
        {
            float[] state = GetState();
            int action = agent.Act(state);
            Debug.LogUpdate(action);
            Cart.Update(action);
            Pole.Update();

            bool done = Cart.MiddlePos.X > Engine.ScreenSize.X / 2 + maxDistanceCar || Cart.MiddlePos.X < Engine.ScreenSize.X / 2 - maxDistanceCar
                || Pole.Pos.Y > Cart.Pos.Y - 200 + 60 || timeStep >= 1000;

            float reward = 1;
            if (done)
                reward = -1;

            states.Add(state);
            actions.Add(action);
            rewards.Add(reward);

            timeStep++;

            if (done)
            {
                if (timeStep > 100)
                    Console.ForegroundColor = ConsoleColor.Green;
                else if (timeStep > 70)
                    Console.ForegroundColor = ConsoleColor.Yellow;
                else
                    Console.ForegroundColor = ConsoleColor.Red;

                Console.WriteLine($"Episode {Main.episode}, Score {timeStep}");
                Console.ForegroundColor = ConsoleColor.Gray;

                Main.episode++;

                episodeLengths.Add(timeStep);
                timeStep = 0;

                Cart.Reset();
                Pole.Reset();

                agent.TrainOneEpisode(states.ToArray(), rewards.ToArray(), actions.ToArray());
                states.Clear();
                actions.Clear();
                rewards.Clear();
                episodeLengths.Clear();
            }
        }

        public void Render()
        {
            Cart.Render();
            Pole.Render();
        }

        public float[] GetState()
        {
            //Debug.LogUpdate("Cart Pos: " + (Cart.MiddlePos.X - Engine.ScreenSize.X / 2), "Cart Velocity: " + Cart.Velocity, "Pole Angle: " + Pole.GetAngle(out _), "Pole gap: " + (Pole.Pos.X - Cart.MiddlePos.X));

            /*Console.WriteLine("Cart Pos: " + Normalize(Cart.MiddlePos.X - Engine.ScreenSize.X / 2, -maxDistanceCar, maxDistanceCar) +
                              "\nCart Velocity: " + Normalize(Cart.Velocity, -20, 20) +
                              "\nPole Angle: " + Normalize(Pole.GetAngle(out _), -0.72f, 0.72f) +
                              "\nPole gap: " + Normalize(Pole.Pos.X - Cart.MiddlePos.X, -130, 130));*/

            return new float[4]
            {
            Normalize(Cart.MiddlePos.X - Engine.ScreenSize.X / 2, -maxDistanceCar, maxDistanceCar),
            Normalize(Cart.Velocity, -20, 20),
            Normalize(Pole.GetAngle(out _), -0.72f, 0.72f),
            Normalize(Pole.Pos.X - Cart.MiddlePos.X, -130, 130)
            };
        }

        public float Normalize(float value, float min, float max)
            => (value - min) / (max - min);
    }
}
